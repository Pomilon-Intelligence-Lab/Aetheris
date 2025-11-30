import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import AetherisConfig
from .expert import Expert

class SparseMoELayer(nn.Module):
    """Memory-optimized Sparse MoE with efficient routing."""
    def __init__(self, config: AetherisConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.load_balancing_coef = config.load_balancing_coef
        self.z_loss_coef = config.router_z_loss_coef

        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config.d_model, config.d_ff)
                                      for _ in range(config.num_experts)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_norm = self.norm(x)
        flat_x = x_norm.view(-1, D)

        # Routing Logits with stability
        gate_logits = self.gate(flat_x)

        # Clamp logits to prevent overflow
        gate_logits = torch.clamp(gate_logits, min=-10.0, max=10.0)

        # Z-Loss for stability
        z_loss = torch.mean(torch.logsumexp(gate_logits, dim=-1)**2) * self.z_loss_coef

        if self.training:
            # Reduce noise for stability
            gate_logits = gate_logits + torch.randn_like(gate_logits) * 1e-3

        gate_probs = F.softmax(gate_logits, dim=-1)
        gate_weights, expert_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # Normalize weights for stability
        gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Load balancing loss
        # Use only the top-1 expert for load balancing calculation to keep it simple and consistent
        expert_mask = F.one_hot(expert_indices[:, 0], num_classes=self.num_experts).float()
        fraction_routed = expert_mask.mean(dim=0)
        mean_prob = gate_probs.mean(dim=0)

        aux_loss = (self.num_experts * torch.sum(fraction_routed * mean_prob)) * self.load_balancing_coef
        total_aux_loss = aux_loss + z_loss

        # Efficient dispatch with in-place operations
        # Accumulate in float32 to prevent overflow during aggregation
        final_output = torch.zeros_like(flat_x, dtype=torch.float32)

        # Iterate over all k selected experts
        for k_idx in range(self.top_k):
            for i, expert in enumerate(self.experts):
                # Find tokens routed to expert 'i' at the k-th position
                mask = (expert_indices[:, k_idx] == i)
                if not mask.any():
                    continue

                expert_input = flat_x[mask]
                expert_out = expert(expert_input)

                # Apply weights
                weights = gate_weights[mask, k_idx].unsqueeze(1)
                
                # Cast to float32 for accumulation
                expert_out = expert_out.to(torch.float32)
                weights = weights.to(torch.float32)

                # Accumulate output (add to existing results from other experts)
                final_output[mask] += expert_out * weights
        
        # Cast back to original dtype
        final_output = final_output.to(flat_x.dtype)

        return x + final_output.view(B, L, D), total_aux_loss
