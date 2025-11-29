import torch
import time
import os
from aetheris.utils import save_checkpoint, load_latest_checkpoint, calculate_model_stats

class Trainer:
    def __init__(self, model, optimizer, scaler, config, device, checkpoint_dir, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger
        
        self.model.to(self.device)

    def validate(self, val_loader, global_step):
        self.model.eval()
        total_loss = 0
        total_items = 0
        num_batches = 100 # Validate on 100 batches to save time
        
        print(f"\n[Validation] Starting validation at step {global_step}...")
        
        with torch.no_grad():
             for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                    
                input_ids, labels = batch
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Auto-cast context
                if self.device.type == 'cuda':
                    autocast_dtype = torch.float16
                else:
                    autocast_dtype = torch.bfloat16
                    
                use_autocast = True if self.config.torch_dtype != torch.float32 else False
                
                if use_autocast:
                    with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=autocast_dtype):
                        output = self.model(input_ids, labels)
                else:
                    output = self.model(input_ids, labels)
                
                total_loss += output["loss"].item()
                total_items += 1
        
        avg_loss = total_loss / total_items if total_items > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"[Validation] Step {global_step} | Loss: {avg_loss:.4f} | PPL: {perplexity:.4f}")
        self.model.train()
        return avg_loss

    def train_epoch(self, train_loader, total_steps, start_step=0, stage_name="Training", val_loader=None, eval_every=500):
        print(f"\n{'='*70}\nStarting {stage_name}: Target Steps={total_steps}\n{'='*70}")
        self.model.train()
        global_step = start_step
        running_loss = 0

        print("Initializing data iterator...")
        train_iter = iter(train_loader)

        print("Fetching first batch...")

        while global_step < total_steps:
            step_start = time.time()

            # Removed periodic cache clearing for performance

            self.optimizer.zero_grad(set_to_none=True)

            try:
                batch = next(train_iter)
                if global_step == start_step:
                    print(f"✓ First batch loaded! Starting training loop...")
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids, labels = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Determine autocast dtype
            if self.device.type == 'cuda':
                autocast_dtype = torch.float16
            else:
                autocast_dtype = torch.bfloat16

            # Check if we should use autocast (skip if model uses float32)
            use_autocast = True
            if self.config.torch_dtype == torch.float32:
                use_autocast = False

            if use_autocast:
                with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', dtype=autocast_dtype):
                    output = self.model(input_ids, labels)
                    loss = output["loss"]
            else:
                output = self.model(input_ids, labels)
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"WARNING: NaN/Inf gradient at step {global_step}, skipping update")
            else:
                self.scaler.step(self.optimizer)

            self.scaler.update()

            global_step += 1
            running_loss += loss.item()

            if global_step % 10 == 0:
                avg_loss = running_loss / 10
                t_diff = time.time() - step_start
                if self.device.type == 'cuda':
                    mem = torch.cuda.memory_allocated() / 1e9
                    max_mem = torch.cuda.max_memory_allocated() / 1e9
                    mem_str = f"VRAM: {mem:.1f}GB (peak: {max_mem:.1f}GB)"
                else:
                    mem_str = "CPU Mode"
                
                tokens_per_sec = (self.config.max_seq_len * input_ids.size(0)) / t_diff
                print(f"  Step {global_step}/{total_steps} | Loss: {avg_loss:.4f} | "
                      f"{mem_str} | {tokens_per_sec:.0f} tok/s")
                running_loss = 0

            if global_step % 500 == 0:
                save_checkpoint(self.model, self.optimizer, self.scaler, global_step, stage_name, self.checkpoint_dir)
                
            if val_loader is not None and global_step % eval_every == 0 and global_step > start_step:
                self.validate(val_loader, global_step)

        return global_step
