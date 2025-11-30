import argparse
import sys
import torch
import os
import torch.nn.functional as F
from aetheris.config import AetherisConfig
from aetheris.model import HybridMambaMoE
from aetheris.data import create_streaming_loader, get_tokenizer
from aetheris.utils import load_latest_checkpoint, calculate_model_stats
from aetheris.trainer import Trainer

def train_command(args):
    print(f"\n{'='*70}")
    print(f"Aetheris Training")
    print(f"Config: {args.config}")
    
    if args.hf_token:
        print(f"Using Hugging Face token: {args.hf_token[:10]}...")
        from huggingface_hub import login
        login(token=args.hf_token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    config = AetherisConfig.from_yaml(args.config)
    tokenizer = get_tokenizer()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model Size: d_model={config.d_model}, layers={config.n_layer}")
    print(f"{'='*70}\n")

    model = HybridMambaMoE(config).to(device)

    # Apply weight initialization
    print("Applying proper weight initialization...")
    model.apply(model._init_weights)

    # Calculate model stats
    stats = calculate_model_stats(model)
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Trainable Parameters: {stats['trainable_params']:,}")

    # Use lower learning rate for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                                 betas=(0.9, 0.95), eps=1e-8, fused=False if device.type == 'cpu' else True)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', init_scale=2**10)

    start_step, current_stage = load_latest_checkpoint(model, optimizer, scaler, device, args.checkpoint_dir, args.checkpoint_name)
    
    trainer = Trainer(model, optimizer, scaler, config, device, args.checkpoint_dir)

    # --- STAGE 1: PRE-TRAINING ---
    if current_stage == "Pre-Training" or start_step == 0:
        pt_loader = create_streaming_loader("cerebras/SlimPajama-627B", "train",
                                           tokenizer, config, args.batch_size, mode="pretrain", 
                                           hf_token=args.hf_token, start_step=start_step)
        
        # Validation loader (no skipping needed, always from start of val set)
        pt_val_loader = create_streaming_loader("cerebras/SlimPajama-627B", "validation",
                                               tokenizer, config, args.batch_size, mode="pretrain", 
                                               hf_token=args.hf_token)

        start_step = trainer.train_epoch(pt_loader, total_steps=args.pretrain_steps, 
                                       start_step=start_step, stage_name="Pre-Training",
                                       val_loader=pt_val_loader)
        current_stage = "SFT"
        start_step = 0

    # --- STAGE 2: SFT ---
    print("\n=== STAGE 2: SFT ===")
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-5

    sft_loader = create_streaming_loader("OpenAssistant/oasst1", "train",
                                        tokenizer, config, args.batch_size, mode="sft", 
                                        hf_token=args.hf_token, start_step=start_step)

    sft_val_loader = create_streaming_loader("OpenAssistant/oasst1", "validation",
                                            tokenizer, config, args.batch_size, mode="sft", 
                                            hf_token=args.hf_token)
    
    trainer.train_epoch(sft_loader, total_steps=args.sft_steps, 
                      start_step=start_step, stage_name="SFT",
                      val_loader=sft_val_loader)

    print("\nTraining Complete!")

@torch.no_grad()
def generate_command(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AetherisConfig.from_yaml(args.config)
    tokenizer = get_tokenizer()

    model = HybridMambaMoE(config).to(device).to(config.torch_dtype)

    load_latest_checkpoint(model, None, None, device, args.checkpoint_dir, args.checkpoint_name)
    model.eval()

    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # --- INFERENCE SANITY CHECK ---
    print(f"\n[SANITY CHECK] Inference Tokenizer: {tokenizer.name_or_path}")
    print(f"[SANITY CHECK] Vocab Size: {tokenizer.vocab_size}")
    print(f"[SANITY CHECK] Input IDs: {input_ids.tolist()}")
    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print(f"[SANITY CHECK] Decoded Prompt: '{decoded_prompt}'\n")
    # ------------------------------
    
    generated_ids = input_ids.clone()
    history_ids = set(input_ids[0].tolist())

    print("-" * 50)
    print(f"Prompt: {prompt}")
    print("Generated Continuation:")

    # Start generation loop
    for step in range(max_new_tokens):
        # Check if we should use autocast (skip if model uses float32)
        use_autocast = True
        if config.torch_dtype == torch.float32:
            use_autocast = False
        
        if use_autocast:
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=model.config.torch_dtype):
                outputs = model(generated_ids)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :]
        else:
            outputs = model(generated_ids)
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :]

        # --- DEBUG: Print Top Predictions for First Step ---
        if step == 0:
            probs = F.softmax(next_token_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)
            print("\n[DEBUG] Step 0 Top-5 Predictions:")
            for i in range(5):
                token_idx = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                token_str = tokenizer.decode([token_idx])
                print(f"  {i+1}. '{token_str}' ({prob:.4f})")
            print("-----------------------------------")
        # ---------------------------------------------------

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

        if next_token_item == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        history_ids.add(next_token_item)

        new_token_text = tokenizer.decode(next_token.squeeze().tolist(), skip_special_tokens=True)
        print(new_token_text, end="", flush=True)

    print("\n" + "-" * 50)

def info_command(args):
    config = AetherisConfig.from_yaml(args.config)
    model = HybridMambaMoE(config)
    
    total_params = 0
    dense_params = 0   # Parameters active for EVERY token
    expert_params = 0  # Parameters in all MoE Experts

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if 'experts' in name:
            expert_params += numel
        else:
            dense_params += numel

    single_expert_size = expert_params / config.num_experts if config.num_experts > 0 else 0
    active_per_token_params = dense_params + (single_expert_size * config.top_k)

    def format_count(count):
        return f"{count / 1_000_000:.2f}M"

    print("=" * 50)
    print("Hybrid Mamba-MoE Model Parameter Analysis")
    print("=" * 50)
    print(f"Total Model Layers (N_Layer): {config.n_layer}")
    print(f"MoE Experts per Layer: {config.num_experts}")
    print(f"Active Experts (Top-K): {config.top_k}")
    print("-" * 50)
    print(f"Total Parameters (Checkpoint Size): {format_count(total_params)}")
    print(f"Dense (Always Active) Parameters: {format_count(dense_params)}")
    print(f"Expert-Only Parameters: {format_count(expert_params)}")
    print("-" * 50)
    print(f"**Active Parameters (Per-Token Compute Load): {format_count(active_per_token_params)}**")
    print(" (This is the 'Dense' parameters + the K active expert parameters)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Aetheris CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    train_parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), help="HuggingFace Token")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    train_parser.add_argument("--pretrain_steps", type=int, default=50000, help="Number of pretraining steps")
    train_parser.add_argument("--sft_steps", type=int, default=1000, help="Number of SFT steps")
    train_parser.add_argument("--checkpoint_name", type=str, default="checkpoint_current.pth", help="Checkpoint file name to load from")

    # Generate Command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    gen_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory with checkpoints")
    gen_parser.add_argument("--checkpoint_name", type=str, default="checkpoint_current.pth", help="Checkpoint file name")
    gen_parser.add_argument("--prompt", type=str, default="The quick brown fox", help="Prompt for generation")
    gen_parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    gen_parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling")
    gen_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--repetition_penalty", type=float, default=3.0, help="Repetition penalty")

    # Serve Command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    serve_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory with checkpoints")
    serve_parser.add_argument("--checkpoint_name", type=str, default="checkpoint_current.pth", help="Checkpoint file name")


    # Info Command
    info_parser = subparsers.add_parser("info", help="Show model info")
    info_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "serve":
        import uvicorn
        from aetheris.api.server import app, get_engine
        
        # Initialize engine before starting server
        engine = get_engine()
        # You might want to pass config/checkpoint paths to get_engine here if it supported arguments
        # For now, it defaults or we need to modify get_engine or InferenceEngine to take args.
        # But `get_engine` is a simple global accessor. 
        # Better: Initialize a global engine with args here.
        from aetheris.inference import InferenceEngine
        import aetheris.api.server
        
        aetheris.api.server.engine = InferenceEngine(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name=args.checkpoint_name
        )
        
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "info":
        info_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
