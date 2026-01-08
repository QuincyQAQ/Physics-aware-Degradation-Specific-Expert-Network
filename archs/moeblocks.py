import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .arch_model import NAFBlock
except:
    from arch_model import NAFBlock

class MoEBlock(nn.Module):
    def __init__(self, c, n=5, used=3):
        super().__init__()
        self.used = int(used)
        self.num_experts = n
        self.experts = nn.ModuleList([NAFBlock(c=c) for _ in range(n)])

    # Sparse implementation for large n
    def forward(self, feat, weights):
        B, _, _, _ = feat.shape
        k = self.used
        # Get top-k weights and indices
        topk_weights, topk_indices = torch.topk(weights, k, dim=1)  # (B, k)
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
        # Apply l1 normalization to keep the sum to 1 and maintain aspect relation between weights
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)  # (B, k)
        mask = torch.zeros(B, self.num_experts, dtype=torch.float32, device=feat.device)
        mask.scatter_(1, topk_indices, 1.0)  # Set 1.0 for used experts
        
        # Initialize output tensor
        outputs = torch.zeros_like(feat)
        
        # Process only used experts
        for expert_idx in range(self.num_experts):
            batch_mask = mask[:, expert_idx].bool()  # Convert to boolean mask
            if batch_mask.any():
                # Get the weights for this expert
                expert_weights = topk_weights[batch_mask, (topk_indices[batch_mask] == expert_idx).nonzero()[:, 1]]
                expert_out = self.experts[expert_idx](feat[batch_mask])
                outputs[batch_mask] += expert_out * expert_weights.view(-1, 1, 1, 1)
        
        return outputs, expert_counts, weights

# 
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    net  = MoEBlock(c = img_channel, 
                            n=5,
                            used=3)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output = net(torch.randn((4, 3, 256, 256)), F.softmax(torch.randn((4,5))))
