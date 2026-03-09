# Project skills & conventions

**Read this file before writing or modifying code in this project.**

---

## 1. Project layout

- **Shell scripts**: `commands/` — all `.sh` files live here; run from repo root.
- **Plots**: `plots/` — all saved figures go here.
- **Results**: `results/` — JSON data files from experiment runs.
- Do not scatter outputs across ad-hoc directories; use the standard folders above.

## 2. Coding style

- **Types**: Use type hints where helpful. Python 3.11+ syntax is fine (`list[int]`, `X | None`).
- **Paths**: Use `Path(__file__).resolve().parent` for script-relative paths; avoid hardcoded absolute paths.
- **Clarity**: Precise implementation; brief docstrings where they help. No unnecessary verbosity.
- **Efficiency**: Avoid frequent GPU–CPU sync or extra copies; keep hot paths efficient.
- **YAGNI**: Don't add defensive code for edge cases not asked for.
- **Execution**: Always use `uv run` instead of raw `python`.

## 3. Experiment methodology

- **Multiple runs**: Always run N ≥ 5 independent models per condition. Report mean ± 95% CI.
- **95% CI**: `1.96 * std / sqrt(n)`. Use this consistently for all error bars.
- **Hyperparameter capture**: Save all hyperparameters to JSON alongside results. Every figure should be reproducible from saved configs and command-line args.
- **Sweeps**: Vary one parameter at a time when possible. Use shell scripts to orchestrate sweeps; save results per condition as separate JSON files.
- **Baselines**: Always include a meaningful baseline (e.g., random init, untrained model) for context.

## 4. Plotting

- **Publication-quality**: Figures should be paper-ready (NeurIPS/ICML style): clear, readable, self-explanatory.
- **Typography**: Sufficiently large font sizes for axis labels (≥13pt), tick labels (≥11pt), title, and legend.
- **Error bars**: Always include 95% CI error bars when reporting metrics across runs.
- **Labels**: Always set axis labels and a title. Use a legend when multiple series are shown.
- **Format**: Save as **PDF** (`bbox_inches="tight"`) into the `plots/` directory. PDF preserves vector quality for papers.

## 5. Training conventions

- **Optimizer**: AdamW with weight-decay / no-decay groups (bias, layernorm, embeddings → no decay).
- **LR schedule**: Cosine decay with warmup (typically 5% of total steps).
- **Mixed precision**: Use bfloat16 autocast + TF32 for transformer training.
- **torch.compile**: Enable for transformers when training time exceeds compilation overhead.
- **Logging**: Use wandb for online tracking. Also write a local log file as backup.
- **Checkpointing**: Save periodically; include both model and optimizer state for resumption.

## 6. JAX conventions

- JAX uses a functional programming model. All functions passed to `jit`/`vmap`/`grad` must be **pure** (no side effects, no external mutable state).
- Don't use Python iterators inside JIT-traced code; use `jax.lax.fori_loop` or `jax.lax.scan`.
- For debug printing inside JIT, use `jax.debug.print()`.
- `jax.jit`, `jax.vmap`, `jax.grad` require static shapes. Design experiments to avoid dynamic shapes.
