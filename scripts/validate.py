import argparse
import os
import torch
import math
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from aetheris.config import AetherisConfig
from aetheris.model import HybridMambaMoE
from aetheris.data import create_streaming_loader, get_tokenizer
from aetheris.utils import load_latest_checkpoint

@torch.no_grad()
def evaluate_model(model, val_loader, device, max_batches=100):
    print(f"\n{'='*50}\nStarting Validation (Max {max_batches} batches)\n{'='*50}")

    model.eval()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch in val_loader:
        if num_batches >= max_batches:
            break

        input_ids, labels = batch
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            output = model(input_ids, labels)
            loss = output["loss"]

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 20 == 0:
            print(f"-> Processed {num_batches}/{max_batches} batches...")

    end_time = time.time()

    if num_batches == 0:
        print("No validation batches were processed.")
        return float('inf')

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"\n--- Validation Results ---")
    print(f"Total batches processed: {num_batches}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"--------------------------\n")

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Validate Aetheris Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory with checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_current.pth", help="Checkpoint file name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace Token")
    parser.add_argument("--dataset", type=str, default="cerebras/SlimPajama-627B", help="Dataset to validate on")
    parser.add_argument("--dataset_mode", type=str, default="pretrain", help="pretrain or sft")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AetherisConfig.from_yaml(args.config)
    tokenizer = get_tokenizer()

    model = HybridMambaMoE(config).to(device).to(config.torch_dtype)

    load_latest_checkpoint(model, None, None, device, args.checkpoint_dir, args.checkpoint_name)
    
    val_loader = create_streaming_loader(args.dataset, "validation", tokenizer, config, args.batch_size, mode=args.dataset_mode, hf_token=args.hf_token)

    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    main()
