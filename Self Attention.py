import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim):
        """
        Initialize the self-attention layer.

        Args:
            dim (int): The input and output dimensionality of the model.
        """
        super().__init__()
        self.dim = dim

        # Linear layers to project input X into queries, keys, and values.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Dropout applied to attention weights to prevent overfitting.
        self.dropout = nn.Dropout(0.1)

        # Final linear projection after attention computation.
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        """
        Forward pass for self-attention.

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            attention_mask (Tensor, optional): Binary mask of shape (batch_size, seq_len, seq_len),
                                               where 0 indicates positions to mask (ignore).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        # Compute query, key, and value matrices
        Q = self.q_proj(X)  # (B, L, D)
        K = self.k_proj(X)  # (B, L, D)
        V = self.v_proj(X)  # (B, L, D)

        # Scaled dot-product attention scores
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)  # (B, L, L)

        # Apply attention mask (if provided)
        if attention_mask is not None:
            # Replace masked positions with a large negative value
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Normalize scores with softmax to obtain attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (B, L, L)

        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)

        # Compute the final attention output
        context = attn_weights @ V  # (B, L, D)

        # Project the output back to original dimension
        return self.out_proj(context)

# Example input: batch_size=3, seq_len=4, dim=4
X = torch.rand(3, 4, 4)

# Example attention mask (1 = attend, 0 = ignore)
mask = torch.tensor([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
])  # (3, 4)

# Expand to (B, L, L) format for masking attention weights
attention_mask = mask.unsqueeze(1).repeat(1, 4, 1)

# Instantiate and run self-attention
model = SelfAttention(dim=4)
output = model(X, attention_mask)
print(output.shape)  # Expected: (3, 4, 4)
