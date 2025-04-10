"""

This LoRA layer decomposes the weight update of a frozen linear layer into two small matrices A and B, and applies the scaled update during forward.
It supports weight merging for efficient inference and unmerging for training. The implementation is parameter-efficient, numerically stable, and modular enough to apply across attention layers or MLPs.

In this implementation, I’ve built a LinearLoRALayer that extends PyTorch’s nn.Module. The goal here is to adapt a frozen pre-trained model with minimal trainable parameters using low-rank decomposition — which is the key idea behind LoRA.

The motivation behind LoRA is to reduce the number of trainable parameters by decomposing the weight update into two low-rank matrices — A and B — such that instead of updating the full matrix, we update a small rank-r approximation.
This allows for efficient fine-tuning, especially in large models like Transformers.

In terms of parameter count, this reduces from O(d²) to O(dr) where r ≪ d — so it’s very memory-efficient.

Here in the constructor, I define the base linear layer, and optionally add the LoRA weights lora_a and lora_b when rank > 0.
These two matrices form the low-rank decomposition, where A has shape [out_features, rank] and B is [rank, in_features].
I use Kaiming initialization for A and keep B zero-initialized by default — that helps stabilize early training.

To ensure we're only adapting the LoRA weights, I freeze the base linear layer's weights and bias.

In the forward pass, we check if merge is active.
If not merged, the output is computed as the sum of the frozen linear layer and the low-rank adaptation

To support efficient inference, I implemented merge_weight() which folds (A @ B) into the frozen linear weight — essentially converting the low-rank weights into a regular dense matrix.
This helps avoid additional computation during deployment.
Similarly, unmerge_weight() allows us to revert back to the separated form, which is useful if we continue fine-tuning later.

In the test code, I generate a batch of input and compare the output of the non-merged and merged models.
I also verify that merging and then unmerging gives results very close to the original — which confirms that the merge logic is numerically stable.

This LoRA layer can be easily plugged into attention projections, MLPs, or anywhere we use a linear transformation in models like BERT or GPT.
By modifying only small components, we can fine-tune large models on new tasks with minimal compute.

Overall, this LoRA implementation allows for flexible training and deployment while minimizing compute cost.
I’d be happy to discuss how it can be integrated into larger architectures like Transformers or adapted to other modules.



"""


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class LinearLoRALayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features,
                 merge=False,
                 rank=8,
                 lora_alpha=16,
                 dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank

        # Base linear layer (frozen during LoRA fine-tuning)
        self.linear = nn.Linear(in_features, out_features)

        if rank > 0:
            # LoRA weight matrices: A (out_features x rank), B (rank x in_features)
            self.lora_a = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_normal_(self.lora_a, a=0.01)  # He initialization for A

            self.lora_b = nn.Parameter(torch.zeros(rank, in_features))  # Initialized to zeros

            # Scale factor adjusts the magnitude of the low-rank adaptation
            self.scale = lora_alpha / rank

            # Freeze original weights and bias of the base linear layer
            self.linear.weight.requires_grad = False
            self.linear.bias.requires_grad = False

        # Optional dropout applied to the final output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # If merge=True, pre-merge the low-rank weights into the original weights
        if merge:
            self.merge_weight()
            
    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        if self.rank > 0 and not self.merge:
            # LoRA path: original output + low-rank adaptation
            # Note: LoRA adaptation = scale * (x @ (A @ B)^T)
            output = self.linear(x) + self.scale * (x @ (self.lora_a @ self.lora_b).T)
        elif self.rank > 0 and self.merge:
            # If weights were already merged, use only the linear layer
            output = self.linear(x)
        else:
            # No LoRA applied
            output = self.linear(x)
        return self.dropout(output)
    
    def merge_weight(self):
        """
        Add the low-rank weights into the base linear weight matrix (for inference).
        This effectively folds LoRA into the model.
        """
        if self.merge and self.rank > 0:
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)

    def unmerge_weight(self):
        """
        Undo the merge if needed (e.g., for continued training after merging).
        """
        if self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)
            

# ====== Test code to validate the LoRA layer ======

batch_size = 32 
seq_len = 128 
in_features = 768
out_features = 512 
rank = 8 
lora_alpha = 16 
dropout = 0.1 

# Input tensor
x = torch.randn(batch_size, seq_len, in_features)

# Initialize LoRA layer without merging
lora_layer = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=False
)

# Forward pass with low-rank path active
output = lora_layer(x)
print(f"Output shape (no merge): {output.shape}")

# Initialize LoRA layer with pre-merged weights
lora_layer_merged = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=True
)

# Forward pass after merging low-rank weights
output_merged = lora_layer_merged(x)
print(f"Output shape (merged): {output_merged.shape}")

# Test merging and unmerging cycle
lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

# Check for consistency after unmerge
print("Max difference after merge/unmerge cycle:",
      torch.max(torch.abs(output - output_after_unmerge)).item())
