"""
Transformer language model training (custom loop).

Pre-tokenized data → GPT-2 model → next-token prediction with
cosine LR schedule, wandb logging, periodic validation, and checkpointing.

Adapt by modifying: model config (n_embd, n_layer, etc.), data paths,
hyperparameters, or the training loop itself.
"""

import argparse
import os
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb


# ── Arguments ────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer LM training (custom loop)")

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
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=8)

    # Logging & checkpointing
    p.add_argument("--log_file", type=str, default="logs/training.log")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--val_steps", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=50000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Data (required — no hardcoded paths)
    p.add_argument("--tokenizer_path", type=str, required=True,
                   help="Path to pretrained tokenizer directory")
    p.add_argument("--train_path", type=str, required=True,
                   help="Path to pre-tokenized training data (.npy)")
    p.add_argument("--val_path", type=str, required=True,
                   help="Path to pre-tokenized validation data (.npy)")

    # Device
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────


def load_data(args) -> tuple[DataLoader, DataLoader]:
    """Load pre-tokenized numpy arrays into DataLoaders."""
    train_data = np.load(args.train_path)
    val_data = np.load(args.val_path)
    print(f"Training samples: {len(train_data):,}  |  Validation samples: {len(val_data):,}")

    common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


# ── Model ────────────────────────────────────────────────────────────────────


def create_model(args, tokenizer: AutoTokenizer) -> nn.Module:
    """Create a GPT-2 model from config and compile it."""
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
    assert tokenizer.vocab_size == args.vocab_size, \
        f"Tokenizer vocab ({tokenizer.vocab_size}) != args.vocab_size ({args.vocab_size})"

    print(f"Model config: {config}")
    model = AutoModelForCausalLM.from_config(config).to(args.device)

    print("Compiling model (may take ~60s on first run)...")
    model = torch.compile(model)
    return model


# ── Optimizer & scheduler ────────────────────────────────────────────────────


def setup_optimizer(model: nn.Module, total_steps: int, args):
    """AdamW with weight-decay / no-decay parameter groups + cosine schedule."""
    no_decay_keywords = ("bias", "ln", "layernorm", "norm", "embedding", "emb")
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
    )

    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Cosine schedule: {total_steps} total steps, {warmup_steps} warmup steps")
    return optimizer, scheduler


# ── Utilities ────────────────────────────────────────────────────────────────


def compute_grad_norm(params: Iterator[nn.Parameter]) -> float:
    total = sum(p.grad.detach().norm(2).item() ** 2 for p in params if p.grad is not None)
    return total ** 0.5


def save_checkpoint(model, optimizer, step: int, args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    print(f"Checkpoint → {path}")


# ── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, pad_id: int,
             device: str, max_batches: int) -> float:
    """Compute average validation loss over max_batches."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_loss, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ids = batch.to(device, non_blocking=True)
        logits = model(input_ids=ids[:, :-1]).logits
        B, S, V = logits.shape
        total_loss += loss_fn(logits.reshape(B * S, V), ids[:, 1:].reshape(B * S)).item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ── Training loop ────────────────────────────────────────────────────────────


def train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, args):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    log_f = open(args.log_file, "w")
    dtype_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    step = 0
    for epoch in range(args.num_epochs):
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            ids = batch.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with dtype_ctx:
                logits = model(input_ids=ids[:, :-1]).logits
                B, S, V = logits.shape
                loss = loss_fn(logits.reshape(B * S, V), ids[:, 1:].reshape(B * S))

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_steps == 0:
                grad_norm = compute_grad_norm(model.parameters())
                lr = optimizer.param_groups[0]["lr"]
                loss_val = loss.item()
                wandb.log({"train_loss": loss_val, "lr": lr, "grad_norm": grad_norm, "step": step})
                log_f.write(f"step {step}: loss={loss_val:.4f} lr={lr:.6f}\n")
                log_f.flush()

            if step % args.val_steps == 0:
                val_loss = validate(model, val_loader, tokenizer.pad_token_id,
                                    args.device, args.val_batches)
                wandb.log({"val_loss": val_loss, "step": step})
                tqdm.write(f"step {step}: val_loss={val_loss:.4f}")

            if step % args.save_steps == 0:
                save_checkpoint(model, optimizer, step, args)

    log_f.close()
    print("Training complete.")


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    args = get_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    wandb.init(
        project="transformer",
        name=f"custom_lr{args.lr}_bs{args.batch_size}",
        config=vars(args),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    train_loader, val_loader = load_data(args)
    total_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = setup_optimizer(model, total_steps, args)

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    train(model, train_loader, val_loader, optimizer, scheduler, tokenizer, args)
