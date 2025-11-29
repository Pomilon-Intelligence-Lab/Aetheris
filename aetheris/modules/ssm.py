import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import AetherisConfig

def selective_scan_native(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                         B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Memory-efficient scan with reduced intermediate tensors."""
    B_size, L, D_inner = u.shape
    D_state = A.shape[-1]

    # Use in-place operations where possible
    h = torch.zeros(B_size, D_inner, D_state, device=u.device, dtype=u.dtype)
    ys = []

    for l in range(L):
        dt = delta[:, l, :].unsqueeze(-1)
        dA = torch.exp(dt * A)

        B_l = B[:, l, :].unsqueeze(1)
        dB = dt * B_l

        u_t = u[:, l, :].unsqueeze(-1)
        h = dA * h + dB * u_t

        C_l = C[:, l, :].unsqueeze(1)
        y_t = torch.sum(h * C_l, dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)
    return y + u * D

class SSMBlock(nn.Module):
    """Memory-optimized State Space Model with stability improvements."""
    def __init__(self, config: AetherisConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.ssm_d_state
        self.d_inner = config.d_inner

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.conv_d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3,
                        padding=2, groups=self.d_inner, bias=False)
        self.gate_proj = nn.Linear(self.d_model, self.d_inner, bias=False)

        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # Initialize A to be more stable (closer to -1)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.1 - 4.0)
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)

        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(config.d_model)

        # Proper initialization
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.delta_proj.weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_norm = self.norm(x)

        xz = self.in_proj(x_norm)
        x_in, z_gate = xz.chunk(2, dim=-1)
        x_conv = self.conv_d(x_in.transpose(1, 2))
        # Slice off the last 2 elements (the "future" leakage)
        x_conv = x_conv[:, :, :-2].transpose(1, 2)
        x_conv = self.act(x_conv)

        # Add small epsilon to prevent numerical issues and clamp max value
        delta = torch.clamp(F.softplus(self.delta_proj(x_conv)), max=5.0) + 1e-4
        B_ssm = self.B_proj(x_conv)
        C_ssm = self.C_proj(x_conv)

        # Clamp A to prevent extreme values
        A_fixed = -torch.exp(torch.clamp(self.A_log, min=-10.0, max=2.0))
        A_batched = A_fixed.unsqueeze(0).expand(B, -1, -1)

        y_ssm = selective_scan_native(x_conv, delta, A_batched, B_ssm, C_ssm, self.D)

        y_gate = F.silu(self.gate_proj(x_norm)) * y_ssm
        output = self.out_proj(y_gate)

        return x + output
