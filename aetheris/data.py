import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from typing import Dict, Iterator
import os

def get_tokenizer(model_name: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_seq_len, mode="pretrain", buffer_size=500, skip_samples=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.buffer_size = buffer_size
        self.skip_samples = skip_samples

    def _prepare_sft_text(self, example):
        if 'messages' in example:
            text = ""
            for msg in example['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'assistant':
                    text += f"Assistant: {content}{self.tokenizer.eos_token}"
                else:
                    text += f"User: {content}\n"
            return text
        elif 'text' in example:
            return example['text']
        else:
            return ""

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        iterator = iter(self.dataset)
        buffer = []
        
        # Calculate roughly how many items to skip if they were yielded
        # We process skipping in the yield loop
        
        for example in iterator:
            text = (example.get('text', '') if self.mode == "pretrain"
                   else self._prepare_sft_text(example))

            if len(text) < 10:
                continue

            enc = self.tokenizer(text, truncation=True, max_length=self.max_seq_len,
                               return_tensors="pt")
            input_ids = enc['input_ids'][0]

            if len(input_ids) < 2:
                continue

            if len(input_ids) < self.max_seq_len:
                pad_len = self.max_seq_len - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])

            labels = input_ids.clone()
            if len(input_ids) < self.max_seq_len:
                labels[-pad_len:] = -100

            buffer.append((input_ids, labels))

            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for _ in range(self.buffer_size // 2):
                    item = buffer.pop()
                    if self.skip_samples > 0:
                        self.skip_samples -= 1
                        continue
                    yield item

        # Yield remaining
        random.shuffle(buffer)
        while buffer:
            item = buffer.pop()
            if self.skip_samples > 0:
                self.skip_samples -= 1
                continue
            yield item

def create_streaming_loader(dataset_name, split, tokenizer, config, batch_size, mode="pretrain", hf_token=None, start_step=0):
    raw_dataset = load_dataset(dataset_name, split=split, streaming=True,
                              trust_remote_code=True, token=hf_token)
    
    # Calculate samples to skip: start_step * batch_size
    skip_samples = start_step * batch_size
    if skip_samples > 0:
        print(f"  [Loader] Resuming: Fast-forwarding dataset by {skip_samples} samples...")
        
    stream_ds = StreamingDataset(raw_dataset, tokenizer, config.max_seq_len, mode=mode, skip_samples=skip_samples)
    
    # Increase num_workers for better utilization
    return DataLoader(stream_ds, batch_size=batch_size, pin_memory=True,
                     num_workers=4, prefetch_factor=4)
