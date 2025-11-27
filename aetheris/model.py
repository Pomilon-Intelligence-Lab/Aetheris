import torch
import torch.nn as nn
from typing import Dict, Any, List
from .config import AetherisConfig
from .modules import SSMBlock, SparseMoELayer

class HybridMambaMoE(nn.Module):
    def __init__(self, config: AetherisConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            if i % 2 == 0:
                self.layers.append(SSMBlock(config))
            else:
                self.layers.append(SparseMoELayer(config))

        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize embeddings with smaller scale
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def _init_weights(self, module):
        """Apply proper weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, Any]:
        x = self.embedding(input_ids)
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Checkpoint ALL layers for maximum memory savings
                if isinstance(layer, SparseMoELayer):
                    def moe_forward(module, inp):
                        return module(inp)
                    x, aux_loss = torch.utils.checkpoint.checkpoint(
                        moe_forward, layer, x, use_reentrant=False
                    )
                    total_aux_loss = total_aux_loss + aux_loss
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
            else:
                if isinstance(layer, SparseMoELayer):
                    x, aux_loss = layer(x)
                    total_aux_loss = total_aux_loss + aux_loss
                else:
                    x = layer(x)

            # Add gradient clipping per layer to catch issues early
            if self.training and torch.isnan(x).any():
                print(f"WARNING: NaN detected in layer output!")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = self.loss_fn(shift_logits.view(-1, self.config.vocab_size),
                                  shift_labels.view(-1))

            # Scale down aux loss to prevent it from dominating
            total_loss = ce_loss + 0.01 * total_aux_loss

            return {
                "loss": total_loss,
                "ce_loss": ce_loss,
                "aux_loss": total_aux_loss,
                "logits": logits
            }

        return {"logits": logits}
