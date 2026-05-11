# ScatterMoE TinyStories training script.
# Uses scattermoe's Triton kernels for routing and expert computation.
#
# Example:
# uv run scatter_moetinystories.py --num_local_experts 128 --intermediate_size 32 --num_experts_per_tok 4
# uv run scatter_moetinystories.py --low_rank --r 8 --num_local_experts 128 --intermediate_size 32 --num_experts_per_tok 4

import argparse
import json
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, TrainerCallback,
    DataCollatorForLanguageModeling, MixtralConfig,
)

from utils.dataset_utils import TinyStoriesDataset
from utils.moe_utils import MoETrainer, replace_moe_with_scatter, patch_forward_with_aux_loss


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------

def get_model_config(args) -> MixtralConfig:
    return MixtralConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        num_local_experts=args.num_local_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        use_cache=False,
        initializer_range=0.02,
        rope_theta=10000.0,
        seed=args.seed,
        router_aux_loss_coef=args.router_aux_loss_coef,
        output_router_logits=False,  # We compute aux_loss in our router
    )


# ---------------------------------------------------------------------------
# Callbacks (same as moe_tinystories.py)
# ---------------------------------------------------------------------------

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, val_dataset, eval_steps, batch_size=32, num_eval_samples=2000):
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        n = min(num_eval_samples, len(val_dataset))
        self.input_ids = torch.stack(
            [val_dataset[i]['input_ids'] for i in range(n)]
        ).cuda()
        self.n = n

    def _eval_metrics(self, model):
        aux_list = []
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            batch = self.input_ids[start:end]
            outputs = model(batch, labels=batch)
            if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
                aux_list.append(outputs.aux_loss.detach())
        aux_loss = torch.stack(aux_list).mean().item() if aux_list else None
        return aux_loss

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return
        model.eval()
        aux_loss = self._eval_metrics(model)
        model.train()

        log_dict = {}
        if aux_loss is not None:
            log_dict["eval/router_aux_loss"] = aux_loss
        if log_dict:
            wandb.log(log_dict, commit=False)

        if aux_loss is not None:
            print(f"Step {state.global_step}: AuxLoss={aux_loss:.4e}")


class RouterAuxLossAnnealingCallback(TrainerCallback):
    def __init__(self, start: float, end: float, decay_fraction: float):
        self.start = start
        self.end = end
        self.decay_fraction = decay_fraction

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None or state.max_steps <= 0:
            return
        decay_steps = self.decay_fraction * state.max_steps
        if state.global_step >= decay_steps:
            coef = self.end
        else:
            coef = self.start + (self.end - self.start) * (state.global_step / decay_steps)
        model.config.router_aux_loss_coef = coef


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-33M")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--tokenizer_path", type=str, default='/shared_data0/hnwong/cache/tokenizer_tinystories_gpt2_10k')
    parser.add_argument("--train_path", type=str, default='/shared_data0/hnwong/cache/tinystories_train_maxlen_512_dict_10000.npy')
    parser.add_argument("--val_path", type=str, default='/shared_data0/hnwong/cache/tinystories_val_maxlen_512_dict_10000.npy')

    # Architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=32)
    parser.add_argument("--num_local_experts", type=int, default=128)
    parser.add_argument("--num_experts_per_tok", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # Router
    parser.add_argument("--low_rank", action="store_true",
                        help="Use low-rank router instead of dense")
    parser.add_argument("--r", type=int, default=128, help="Rank of the low-rank router")
    parser.add_argument("--router_lr", type=float, default=None,
                        help="Learning rate for router (defaults to --lr if not set)")
    parser.add_argument("--router_init", type=str, default="special_init",
                        choices=["special_init", "standard_init"],
                        help="Low-rank router init: 'special_init' (variance-preserving) or 'standard_init' (same std as dense)")

    # Training schedule
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "constant", "constant_with_warmup", "linear"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--tag", type=str, default="test")

    # Router aux loss
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.001)
    parser.add_argument("--router_aux_loss_anneal", action="store_true")
    parser.add_argument("--router_aux_loss_start", type=float, default=0.01)
    parser.add_argument("--router_aux_loss_decay_fraction", type=float, default=0.2)

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    wandb_name = f"scatter_moe_lr_{args.lr}_bs_{args.batch_size}_wd_{args.weight_decay}_sched_{args.lr_scheduler}_M{args.num_local_experts}_s{args.intermediate_size}_k{args.num_experts_per_tok}_seed_{args.seed}"
    if args.low_rank:
        wandb_name = 'low_rank_r' + str(args.r) + '_' + wandb_name
    wandb.init(project="transformer", name=wandb_name, tags=[args.tag], config=args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path else args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = get_model_config(args)
    print("Using ScatterMoE with low-rank router." if args.low_rank else "Using ScatterMoE.")
    assert tokenizer.vocab_size == args.vocab_size

    model = AutoModelForCausalLM.from_config(config)

    # Replace HF MoE blocks with ScatterMoE blocks
    replace_moe_with_scatter(model, config, low_rank=args.low_rank, r=args.r, router_init=args.router_init)
    # Patch forward to collect aux_loss from ScatterMoE routers
    patch_forward_with_aux_loss(model)

    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    train_dataset = TinyStoriesDataset(args.train_path, max_length=args.max_length)
    val_dataset = TinyStoriesDataset(args.val_path, max_length=args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="no",
        bf16=True,
        tf32=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to="wandb",
        dataloader_persistent_workers=True,
        max_grad_norm=args.max_grad_norm,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    gen_eval_callback = GenerationEvalCallback(
        val_dataset, args.eval_steps, batch_size=32,
    )
    callbacks_list = [gen_eval_callback]
    if args.router_aux_loss_anneal:
        callbacks_list.append(RouterAuxLossAnnealingCallback(
            start=args.router_aux_loss_start,
            end=args.router_aux_loss_coef,
            decay_fraction=args.router_aux_loss_decay_fraction,
        ))

    if args.router_lr is not None:
        print(f"Using MoETrainer with router lr={args.router_lr}")
        trainer = MoETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            router_lr=args.router_lr,
            callbacks=callbacks_list,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks_list,
        )

    trainer.train()
    print("Done!")
