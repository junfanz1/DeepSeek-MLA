"""
GQA reduces the number of key-value heads, which directly reduces memory and compute cost — especially beneficial when scaling models like GPT-4. It strikes a balance between full multi-head attention and the extreme sharing in multi-query attention (MQA).

We use repeat_interleave to replicate key/value heads so that each query head has a matching K/V for attention. This ensures shape consistency while preserving the memory advantage of fewer K/V heads.

We could further extend this module by adding causal masks, rotary position embeddings, or FlashAttention for speedup.

This GroupQueryAttention module is a memory-efficient alternative to standard multi-head attention, allowing us to scale models without quadratic growth in memory, and it's already being used in production LLMs like Mistral and GPT-4.
"""

import torch
import torch.nn as nn
import math

"""
This is an implementation of Group Query Attention, a mechanism adopted by large-scale LLMs such as GPT-4 and Mistral.
The key idea is that multiple query heads share a smaller number of key and value heads, reducing memory and compute cost without significant performance loss.

The input X is projected into queries, keys, and values using linear layers.
We generate num_heads queries but only num_key_value_heads key/value heads.
Then, I use .repeat_interleave() to replicate K/V heads so that every query head has a corresponding K/V for attention, while maintaining compute efficiency.

After computing the dot product of Q and K, we scale by sqrt of the head dimension and apply softmax to get attention weights.
We then weight the values and project the result back to the original hidden dimension via o_proj.

This structure retains the expressiveness of multi-head attention while reducing memory overhead by sharing key/value projections.
It’s a very practical design for scaling LLMs. I can also extend this with rotary embeddings, causal masking, or integrate with FlashAttention.
"""

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()

        # Ensure that the number of attention heads divides hidden_dim evenly
        assert hidden_dim % nums_head == 0
        # Each group of query heads shares one key-value head
        assert nums_head % nums_key_value_head == 0

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head  # Dimension per attention head

        # Projection for queries: outputs [nums_head * head_dim]
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        # Shared projections for keys and values: outputs [nums_kv_head * head_dim]
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        # Output projection: combines attention output back to hidden_dim
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        # X shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = X.size()

        # Project input to Q, K, V
        q = self.q_proj(X)  # Shape: (batch, seq_len, nums_head * head_dim)
        k = self.k_proj(X)  # Shape: (batch, seq_len, nums_kv_head * head_dim)
        v = self.v_proj(X)

        # Reshape to separate heads: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim)

        # Transpose to: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Broadcast K, V to match Q head count
        # Repeat key-value heads for each group of queries
        repeat_factor = self.nums_head // self.nums_key_value_head
        k = k.repeat_interleave(repeat_factor, dim=1)  # Shape: (batch, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention
        # Attention scores: (batch, num_heads, seq_len, seq_len)
        attention_scores = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Softmax over keys
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to values
        attention_output = attention_weights @ v  # (batch, num_heads, seq_len, head_dim)

        # Recombine heads: (batch, seq_len, hidden_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        final_output = self.o_proj(attention_output.view(batch_size, seq_len, -1))

        return final_output

# Test input
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(hidden_dim=128, nums_head=8, nums_key_value_head=4)
print(net(x).shape)  # Should output (3, 2, 128)

"""
Group Query Attention is a more efficient form of multi-head attention, where multiple query heads share a smaller number of key and value heads.
In my implementation, I project the input into Q, K, and V. The number of query heads is num_heads, while the number of key-value heads is smaller, num_key_value_heads.

To align them, I use .repeat_interleave() to replicate each K/V head so that they match the number of query heads.
This allows each query head to compute attention independently while reducing memory and computation cost — which is crucial for scaling large models like GPT-4.

Finally, I reshape and recombine the heads and use a linear layer to project back to the hidden dimension.

We repeat K and V across query groups to simulate independent attention per head — but actually, the key/value weights are shared underneath, so it's still efficient.
Most importantly, the number of parameters is reduced significantly, and in practice it doesn't harm performance — that's why Mistral and GPT-4 use it.

To optimize further, I’d consider a few directions:
- Add rotary positional embeddings to improve generalization on long sequences.
- Add causal masks for autoregressive decoding.
- Replace softmax attention with FlashAttention for memory-bandwidth efficiency.
- Use grouped linear projection to avoid repeating K/V explicitly and save memory during inference.
"""

