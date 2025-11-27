from dataclasses import dataclass, field
import yaml
import torch
from typing import Optional

@dataclass
class AetherisConfig:
    # Model dimensions
    vocab_size: int = 50257
    d_model: int = 768
    n_layer: int = 24
    num_experts: int = 4
    top_k: int = 1
    d_ff: int = 2304 # d_model * 3

    # SSM parameters
    ssm_d_state: int = 16
    ssm_expand: int = 2
    d_inner: Optional[int] = None # Will be d_model * ssm_expand if None

    # Training parameters
    load_balancing_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    max_seq_len: int = 512
    dtype: str = "float16" # "float16", "float32", "bfloat16"

    # Optimization settings
    use_cpu_offload: bool = False
    gradient_checkpointing: bool = True
    checkpoint_ssm_layers: bool = True
    use_flash_attention: bool = False

    def __post_init__(self):
        if self.d_inner is None:
            self.d_inner = self.d_model * self.ssm_expand
        if self.d_ff is None:
             self.d_ff = self.d_model * 3

    @property
    def torch_dtype(self):
        if self.dtype == "float16":
            return torch.float16
        elif self.dtype == "float32":
            return torch.float32
        elif self.dtype == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
