import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Memory-efficient Feed-Forward Network expert with proper initialization."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

        # Proper initialization to prevent NaN
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Force float32 for internal computation to prevent overflow in half precision
        x = x.to(torch.float32)
        
        # Cast weights to float32 for calculation
        # This is necessary because the module weights might be float16
        w1_weight = self.w1.weight.to(torch.float32)
        w2_weight = self.w2.weight.to(torch.float32)
        
        h = F.linear(x, w1_weight)
        h = self.act(h)
        out = F.linear(h, w2_weight)
        
        # Clamp to avoid Inf when casting back to float16
        if orig_dtype == torch.float16:
            out = torch.clamp(out, min=-65500.0, max=65500.0)
            
        return out.to(orig_dtype)
