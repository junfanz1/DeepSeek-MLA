import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import math

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768 # or hidden_dim, hidden_size
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    # tiktoken use GPT-2 50257 tokens
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size

        # attention_mask use register_buffer, no need to calculate gradient hence faster
        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(config.block_size, config.block_size),
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) # simpler version of torch.matmul
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        ) / math.sqrt(self.head_size) # hidden_size is head_size due to single head
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(config) for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    # MLP
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # linear (4 -> 8), weight shape 8 * 4, so embedding weight and lm_head weight are shared
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # normal distribution initialize
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)

        # seq length is input max length
        pos_emb = self.position_embedding_table(
            # make sure position encoding and input idx on same device
            torch.arange(seq_len, device=idx.device)
        )
        # why embedding and position can be added?
        x = token_emb + pos_emb # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # shape is (batch, seq_len, vocab_size)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # if seq too long, only take last block_size amount of tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, self.block_size:]
            # prediction
            logits, _ = self(idx_cond)
            # only check last timestamp prediction
            logits = logits[:, -1, :] # become (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # sampling next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # add to seq
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"},
        )[0]
        import json
        self.encoded_data = []
        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        # segment long text to training set
        for i in range(0, len(full_encoded), self.block_size):
            # one more token as target
            chunk = full_encoded[i : i + self.block_size+1]
            # if not long enough, use eos_token in place
            if len(chunk) < self.block_size + 1:
                chunk += [self.eos_token] * (self.block_size - len(chunk) + 1)
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    def encode(self, text):
        return self.enc.encode(text)
    def decode(self, ids):
        return self.enc.decode(ids)

"""
{"text":"blahblah"}
"""

train_dataset = MyDataset('/path/corpus.jsonl')
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)

model = GPT(GPTConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# cosine learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # forward propagation
        logits, loss = model(x, targets=y)
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # adjust learning rate
        scheduler.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return totaal_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

for epoch in range(2):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
    val_loss = eval(model, val_loader, device)
    print(f'Epoch:{epoch}, Train Loss:{train_loss/len(train_loader):.4f}, Val Loss:{val_loss/len(val_loader):.4f}')
    # save model
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }
    # save each epoch model
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
