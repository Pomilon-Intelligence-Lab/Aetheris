import os
import torch
from typing import Tuple

def save_checkpoint(model, optimizer, scaler, step, stage, checkpoint_dir, checkpoint_name="checkpoint_current.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save({
        'step': step,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }, path)
    print(f"    [Checkpoint] Saved at step {step}")

def load_latest_checkpoint(model, optimizer, scaler, device, checkpoint_dir, checkpoint_name="checkpoint_current.pth") -> Tuple[int, str]:
    path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(path):
        return 0, "Pre-Training"

    print(f"    [Checkpoint] Loading from {path}...")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt['step'], ckpt['stage']

def calculate_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'active_params': int(total_params * 0.6), # Approximation
        'sparsity_ratio': 0.6 # Approximation
    }
