import torch
import torch.nn.functional as F
from typing import Optional, List, Generator
from aetheris.config import AetherisConfig
from aetheris.model import HybridMambaMoE
from aetheris.data import get_tokenizer
from aetheris.utils import load_latest_checkpoint

class InferenceEngine:
    def __init__(self, config_path: str = "configs/default.yaml", checkpoint_dir: str = "checkpoints", checkpoint_name: str = "checkpoint_current.pth", device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.config = AetherisConfig.from_yaml(config_path)
        self.tokenizer = get_tokenizer()
        
        self.model = HybridMambaMoE(self.config).to(self.device).to(self.config.torch_dtype)
        
        # Load checkpoint
        # Note: load_latest_checkpoint expects optimizer and scaler, but for inference we can pass None
        load_latest_checkpoint(self.model, None, None, self.device, checkpoint_dir, checkpoint_name)
        self.model.eval()

    def generate(self, 
                 prompt: str, 
                 max_new_tokens: int = 100, 
                 temperature: float = 0.8, 
                 top_k: int = 0, 
                 top_p: float = 0.9, 
                 repetition_penalty: float = 1.0,
                 stream: bool = False) -> Generator[str, None, None] | str:
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        history_ids = set(input_ids[0].tolist())

        def token_generator():
            nonlocal generated_ids
            for _ in range(max_new_tokens):
                 # Check if we should use autocast (skip if model uses float32)
                use_autocast = True
                if self.config.torch_dtype == torch.float32:
                    use_autocast = False
                
                if use_autocast:
                    with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=self.model.config.torch_dtype):
                        outputs = self.model(generated_ids)
                        logits = outputs['logits']
                        next_token_logits = logits[:, -1, :]
                else:
                    outputs = self.model(generated_ids)
                    logits = outputs['logits']
                    next_token_logits = logits[:, -1, :]

                # Repetition penalty
                for token_id in history_ids:
                    if token_id < next_token_logits.size(-1):
                        logit = next_token_logits[0, token_id].item()
                        if logit > 0:
                            next_token_logits[0, token_id] = logit / repetition_penalty
                        else:
                            next_token_logits[0, token_id] = logit * repetition_penalty

                # Temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Top-p / Top-k
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove.unsqueeze(0), float('-inf'))
                elif top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Sample
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                next_token_item = next_token.item()

                if next_token_item == self.tokenizer.eos_token_id:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                history_ids.add(next_token_item)

                new_token_text = self.tokenizer.decode(next_token.squeeze().tolist(), skip_special_tokens=True)
                yield new_token_text
        
        if stream:
            return token_generator()
        else:
            return "".join(list(token_generator()))

    def generate_full(self, 
                 prompt: str, 
                 max_new_tokens: int = 100, 
                 temperature: float = 0.8, 
                 top_k: int = 0, 
                 top_p: float = 0.9, 
                 repetition_penalty: float = 1.0) -> str:
        return self.generate(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stream=False)
