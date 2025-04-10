import math
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings(action="ignore")

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim, nums_head, dropout=0.1):
        super().__init__()
        self.nums_head = nums_head
        self.head_dim = hidden_dim // nums_head
        self.dropout = dropout

        # Post-layernorm architecture: normalization is applied after the residual connection
        self.layernorm_att = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Linear projections for multi-head self-attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(self.dropout)

        # Feed-forward network (FFN)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layernorm_ffn = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.act_fn = nn.ReLU()
        self.drop_ffn = nn.Dropout(self.dropout)

    def attention_output(self, query, key, value, attention_mask=None):
        """
        Compute masked multi-head self-attention output.
        Shapes:
        - query/key/value: (batch, head, seq_len, head_dim)
        - attention_mask: (batch, head, seq_len, seq_len)
        """

        # Compute scaled dot-product attention scores
        key = key.transpose(2, 3)  # (batch, head, head_dim, seq_len)
        att_weight = torch.matmul(query, key) / math.sqrt(self.head_dim)

        # Apply causal (lower-triangular) mask to prevent attending to future positions
        if attention_mask is not None:
            attention_mask = attention_mask.tril()
            att_weight = att_weight.masked_fill(attention_mask == 0, float('-1e20'))
        else:
            attention_mask = torch.ones_like(att_weight).tril()
            att_weight = att_weight.masked_fill(attention_mask == 0, float('-1e20'))

        # Apply softmax to get attention distribution
        att_weight = torch.softmax(att_weight, dim=-1)

        # Dropout on attention weights
        att_weight = self.drop_att(att_weight)

        # Weighted sum of values
        mid_output = torch.matmul(att_weight, value)  # (batch, head, seq_len, head_dim)

        # Rearrange dimensions and merge heads
        mid_output = mid_output.transpose(1, 2).contiguous()  # (batch, seq_len, head, head_dim)
        batch, seq, _, _ = mid_output.size()
        mid_output = mid_output.view(batch, seq, -1)  # (batch, seq_len, hidden_dim)

        # Final linear projection
        output = self.o_proj(mid_output)
        return output 

    def attention_block(self, X, attention_mask=None):
        """
        Multi-head self-attention block with residual connection and post-layernorm.
        """
        batch, seq, _ = X.size()

        # Project input to multi-head Q/K/V
        query = self.q_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)
        key = self.k_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)
        value = self.v_proj(X).view(batch, seq, self.nums_head, -1).transpose(1, 2)

        # Compute attention output
        output = self.attention_output(query, key, value, attention_mask=attention_mask)

        # Add residual and apply layer normalization
        return self.layernorm_att(output + X)

    def ffn_block(self, X):
        """
        Feed-forward network block with residual connection and post-layernorm.
        """
        up = self.act_fn(self.up_proj(X))      # Expand hidden dimension
        down = self.down_proj(up)              # Project back to original dimension
        down = self.drop_ffn(down)
        return self.layernorm_ffn(down + X)    # Residual + layernorm

    def forward(self, X, attention_mask=None):
        """
        Transformer decoder block = Self-attention + FFN.
        """
        att_output = self.attention_block(X, attention_mask=attention_mask)
        ffn_output = self.ffn_block(att_output)
        return ffn_output

# Testing the decoder with dummy input
x = torch.rand(3, 4, 64)  # (batch=3, seq_len=4, hidden_dim=64)
net = SimpleDecoder(64, 8)

# Create an attention mask for causal + padding masking
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
    .unsqueeze(1)  # (batch, 1, seq_len)
    .unsqueeze(2)  # (batch, 1, 1, seq_len)
    .repeat(1, 8, 4, 1)  # (batch, head, seq_len, seq_len)
)

# Run the decoder forward pass
net(x, mask).shape  # Expected output: (3, 4, 64)
