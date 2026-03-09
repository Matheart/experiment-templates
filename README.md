# Experiment Templates

Reusable templates for rigorous empirical ML research.

## Goals

1. **Reproducibility** — All hyperparameters captured via argparse and saved
   to JSON. Results are reproducible from the saved configuration.
2. **Adaptability** — Templates are task-agnostic. Swap datasets, architectures,
   or training procedures with minimal changes.
3. **Statistical rigor (MLP)** — The MLP template runs N independent models
   in parallel and reports mean ± 95% CI with publication-quality error-bar plots.
4. **Clean training infrastructure (Transformer)** — Organized training loops
   with wandb logging, checkpointing, and LR scheduling, ready to adapt to
   new datasets or architectures.

---

## MLP

### Batched parallel training — `mlp_cifar10.py`

Trains N independent MLPs **simultaneously** on a single GPU using batched
linear layers (`einsum` over the model dimension). Far more efficient than
running N separate training jobs for obtaining error bars.

**Key components:**
- `MultiLinear` / `MultiMLP` — batched modules that run N models in parallel
- `BatchedDataLoader` — GPU-resident data with per-model shuffling
- `plot_bar()` / `plot_sweep()` — publication-quality plotting with 95% CI
- JSON result saving with full hyperparameter capture

```bash
uv run mlp_cifar10.py --n_models 10 --epochs 10 --lr 3e-4
```

**Adapting to a new task:**
1. Replace `get_dataset()` to return your data, `input_dim`, and `num_classes`
2. Adjust `--hidden_dims` for your architecture
3. Add experiment conditions to the main block and use `plot_bar()`

### Parallelizing sweeps

Three approaches for running many small experiments efficiently, each suited
to different use cases:

| Approach | When to use | Files |
|---|---|---|
| **PyTorch batched** | All experiments share the same architecture and data shape | `mlp_cifar10.py` (built-in via `MultiMLP`) |
| **JAX vmap** | Large synthetic sweeps where functional transforms shine | `jax_exp/jax_mlp_parallel.py` |
| **Multiprocessing** | Experiments differ in shape (e.g., varying width) and can't be batched | `small_exp/` |

**Guidelines:**
- Keep execution times **uniform** within a parallel batch — mixing width=64
  and width=2048 causes idle GPUs and can make parallelism slower than sequential.
- For multiprocessing, use ~4 workers; too many introduces dispatch overhead.
- For parameter sweeps, put shell scripts in `commands/` and save results to `results/`:

```bash
# commands/sweep_lr.sh
#!/bin/bash
for lr in 1e-4 3e-4 1e-3; do
  uv run mlp_cifar10.py --lr $lr --n_models 10 --epochs 10 \
    --results_dir results/lr_sweep --plot_dir plots
done
```

Then use `plot_sweep()` from `mlp_cifar10.py` to create a comparative figure (saved as PDF in `plots/`).

### JAX reference — `jax_exp/`

- `jax_mlp_exp.py` — single MLP training with configurable hyperparameters
- `jax_mlp_parallel.py` — parallel ablation over learning rates and seeds via `vmap`
- `jax_basics.py` / `jit_basics.py` — JAX API reference examples

```bash
uv run jax_exp/jax_mlp_parallel.py --lr-values 1e-3 1e-4 1e-5 --seed-values 42 123 456
```

---

## Transformer

### Custom training loop — `transformer_lm.py`

Full-control training loop for transformer language models. Uses HuggingFace
model configs with a hand-written loop for maximum flexibility.

- Cosine LR schedule with warmup
- Weight-decay / no-decay parameter groups
- bfloat16 autocast, TF32, `torch.compile`
- wandb logging, periodic validation, checkpointing

```bash
uv run transformer_lm.py \
  --tokenizer_path /path/to/tokenizer \
  --train_path /path/to/train.npy \
  --val_path /path/to/val.npy
```

### HF Trainer — `trainer_lm.py`

Same task as above using the HuggingFace `Trainer` API. Less code, built-in
distributed training support, automatic logging.

```bash
uv run trainer_lm.py \
  --tokenizer_path /path/to/tokenizer \
  --train_path /path/to/train.npy \
  --val_path /path/to/val.npy
```

### Multi-GPU / DDP — `distributed_training.py`

Template for multi-GPU training using PyTorch `DistributedDataParallel`.
Shows process group setup, `DistributedSampler`, and `mp.spawn`.

---

## Techniques

### Single-GPU speedups

1. **TF32 precision** — enable Tensor Cores (almost free):
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```
2. **Mixed-precision (bfloat16)** — wrap the forward pass:
   ```python
   with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
       logits = model(input_ids=ids).logits
   ```
3. **`torch.compile()`** — fuses ops into optimized kernels. Typically 1.5–2x
   speedup. Worth it when training time exceeds the ~60s compilation overhead.
4. **Larger batch size** — better GPU utilization and more accurate gradients.
   Scale LR by `sqrt(batch_scale_factor)`.
5. **Minimize CPU-GPU sync** — avoid `.item()` or `.to(device)` in the hot loop.

### LR scaling rule

Square root scaling: when increasing batch size by factor k, scale LR by
`sqrt(k)`. LR also scales inversely with model dimension.
