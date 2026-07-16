#!/usr/bin/env python3
"""
Train T5 on flat action language with sample printing during validation.
"""

import os
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

SPECIAL_TOKENS = ["<EOS>"]

class FlatActionDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_input_len=256, max_output_len=256):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row['text']
                if 'action_str' in row and row['action_str']:
                    target = row['action_str']
                else:
                    with open(row['action_file'], 'r') as af:
                        target = af.read().strip()
                self.data.append((text, target))
        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, target = self.data[idx]
        enc = self.tokenizer(text, max_length=self.max_input_len,
                             truncation=True, padding='max_length',
                             return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)

        tgt_enc = self.tokenizer(target, max_length=self.max_output_len,
                                 truncation=True, padding='max_length',
                                 return_tensors='pt')
        labels = tgt_enc['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': labels,
            'text': text,
            'target': target
        }

def compute_metrics(preds, targets):
    exact = sum(p == t for p, t in zip(preds, targets)) / len(preds)
    smooth = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([t.split()], p.split(), smoothing_function=smooth) for p, t in zip(preds, targets)]
    return {'exact_match': exact, 'bleu': np.mean(bleu_scores)}

def train(args):
    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no',
                              gradient_accumulation_steps=args.grad_accum)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()   # saves memory

    train_dataset = FlatActionDataset(args.train_csv, tokenizer, args.max_input_len, args.max_output_len)
    val_dataset = FlatActionDataset(args.val_csv, tokenizer, args.max_input_len, args.max_output_len)

    # Limit validation size for speed
    if len(val_dataset) > 500:
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, range(500))
        print("Validation limited to 500 samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_bleu = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(progress):
            with accelerator.accumulate(model):
                outputs = model(input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        sample_texts = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", disable=not accelerator.is_main_process):
                gen_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=args.max_output_len,
                    num_beams=2,
                    early_stopping=True
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(preds)
                all_targets.extend(batch['target'])
                if len(sample_texts) < 3:
                    for i in range(min(3, len(preds))):
                        if len(sample_texts) < 3:
                            sample_texts.append((batch['text'][i], preds[i], batch['target'][i]))

        metrics = compute_metrics(all_preds, all_targets)
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1} - Loss: {epoch_loss/len(train_loader):.4f}, Exact: {metrics['exact_match']:.4f}, BLEU: {metrics['bleu']:.4f}")
            print("--- Sample predictions ---")
            for i, (txt, pred, tgt) in enumerate(sample_texts[:3]):
                print(f"Text {i+1}: {txt[:100]}...")
                print(f"Pred: {pred}")
                print(f"Targ: {tgt}")
                print()
            # Save best model
            if metrics['bleu'] > best_bleu:
                best_bleu = metrics['bleu']
                patience_counter = 0
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(os.path.join(args.output_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                print(f"  New best model saved (BLEU={best_bleu:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(os.path.join(args.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_output_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
