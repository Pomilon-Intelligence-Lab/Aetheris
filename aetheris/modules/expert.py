import torch
import torch.nn as nn

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
        return self.w2(self.act(self.w1(x)))
