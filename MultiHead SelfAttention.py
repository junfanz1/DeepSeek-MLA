import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        """
        Initialize Multi-Head Attention module.

        Args:
            hidden_dim (int): The total dimensionality of the input embeddings.
            nums_head (int): Number of attention heads.
        """
        super().__init__()
        self.nums_head = nums_head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nums_head  # Dimensionality per head

        assert self.head_dim * nums_head == hidden_dim, "hidden_dim must be divisible by nums_head"

        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.att_dropout = nn.Dropout(0.1)

        # Final linear projection after concatenating all heads
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        """
        Forward pass for multi-head self-attention.

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, 1, seq_len, seq_len),
                                               where 0 indicates masked positions.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = X.size()

        # Project input X to queries, keys, and values
        Q = self.q_proj(X)  # (B, L, D)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # Reshape and permute for multi-head attention:
        # From (B, L, D) -> (B, num_heads, L, head_dim)
        Q = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention for each head
        # Q @ K^T -> (B, num_heads, L, L)
        attention_scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Mask shape: (B, 1, L, L) or broadcastable to it
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-1e20'))

        # Softmax over the last dimension (seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.att_dropout(attention_weights)

        # Compute attention-weighted values: (B, num_heads, L, head_dim)
        context = attention_weights @ V

        # Reshape: (B, num_heads, L, head_dim) -> (B, L, num_heads * head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final linear projection
        return self.o_proj(context)

# Simulated attention mask: shape (batch_size=3, 1, seq_len=2, seq_len=2)
attention_mask = torch.tensor([
    [[0, 1],
     [0, 0]],
    [[0, 0],
     [0, 0]],
    [[1, 0],
     [1, 0]]
]).unsqueeze(1)  # Shape: (3, 1, 2, 2)

# Input tensor: shape (batch_size=3, seq_len=2, hidden_dim=128)
x = torch.rand(3, 2, 128)

# Instantiate multi-head attention
net = MultiHeadAttention(hidden_dim=128, nums_head=8)

# Forward pass
output = net(x, attention_mask)
print(output.shape)  # Expected: (3, 2, 128)
