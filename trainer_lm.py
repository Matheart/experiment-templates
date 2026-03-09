"""
Transformer language model training (HuggingFace Trainer).

Same task as transformer_lm.py but uses the HF Trainer API for simpler code
with built-in distributed training, logging, and checkpointing.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb


# ── Dataset ──────────────────────────────────────────────────────────────────


class PreTokenizedDataset(Dataset):
    """Dataset backed by a pre-tokenized .npy file (memory-mapped for efficiency)."""

    def __init__(self, path: str):
        self.data = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.data[idx].astype(np.int64))
        return {"input_ids": ids, "labels": ids.clone()}


# ── Arguments ────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer LM training (HF Trainer)")

    # Model architecture
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_inner", type=int, default=3072)

    # Training
    p.add_argument("--lr", type=float, default=1.696e-3)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--num_workers", type=int, default=4)

    # Logging & saving
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=50000)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--output_dir", type=str, default="./results")

    # Data (required — no hardcoded paths)
    p.add_argument("--tokenizer_path", type=str, required=True,
                   help="Path to pretrained tokenizer directory")
    p.add_argument("--train_path", type=str, required=True,
                   help="Path to pre-tokenized training data (.npy)")
    p.add_argument("--val_path", type=str, required=True,
                   help="Path to pre-tokenized validation data (.npy)")

    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = get_args()

    wandb.init(
        project="transformer",
        name=f"trainer_lr{args.lr}_bs{args.batch_size}",
        config=vars(args),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.vocab_size == args.vocab_size, \
        f"Tokenizer vocab ({tokenizer.vocab_size}) != args.vocab_size ({args.vocab_size})"

    # Model
    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.max_length,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation="sdpa",
    )
    model = AutoModelForCausalLM.from_config(config)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Data
    train_dataset = PreTokenizedDataset(args.train_path)
    val_dataset = PreTokenizedDataset(args.val_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        tf32=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Training complete.")
