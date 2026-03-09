"""
MLP training template with parallel multi-run error bars.

Trains N independent MLPs simultaneously on CIFAR-10 using batched
linear layers (einsum over the model dimension). All models share data
in GPU memory efficiently via tensor.expand() and produce per-model
metrics for 95% confidence intervals.

To adapt:
  - Dataset: replace get_dataset() and adjust input_dim / num_classes
  - Architecture: change --hidden_dims
  - Experiment: add conditions to the main block and use plot_bar()
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms

DEVICE = "cuda" if t.cuda.is_available() else "cpu"


# ── Batched modules ──────────────────────────────────────────────────────────


class MultiLinear(nn.Module):
    """nn.Linear with a leading model dimension for n_models independent copies.

    weight shape: [n_models, d_out, d_in]
    forward:      [n_models, batch, d_in] -> [n_models, batch, d_out]
    """

    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]


class MultiMLP(nn.Module):
    """Batched MLP: n_models independent MLPs run in parallel."""

    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
            layers.append(MultiLinear(n_models, d_in, d_out))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.flatten(2))


# ── GPU-resident dataloader ──────────────────────────────────────────────────


class BatchedDataLoader:
    """Preloaded GPU dataloader with per-model independent shuffling.

    Data lives on GPU once; each of the M models gets its own random
    permutation of sample indices per epoch.
    """

    def __init__(self, x: t.Tensor, y: t.Tensor, batch_size: int, shuffle: bool = True):
        self.x, self.y = x, y
        self.M, self.N = x.shape[:2]
        self.bs, self.shuffle = batch_size, shuffle

    def _perm(self):
        base = t.arange(self.N, device=self.x.device)
        if self.shuffle:
            return t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
        return base.expand(self.M, -1)

    def __iter__(self):
        self.perm = self._perm()
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        bx = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)])
        by = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)])
        return bx, by

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ── Dataset ──────────────────────────────────────────────────────────────────


def get_dataset():
    """Returns (train_dataset, test_dataset, input_dim, num_classes).
    Replace this function to adapt the template to a different task."""
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    root = os.path.join(str(Path.home()), ".pytorch", "CIFAR10")
    train = datasets.CIFAR10(root, download=True, train=True, transform=tfm)
    test = datasets.CIFAR10(root, download=True, train=False, transform=tfm)
    return train, test, 3 * 32 * 32, 10


def to_tensors(ds):
    xs, ys = zip(*ds)
    return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)


# ── Training & evaluation ────────────────────────────────────────────────────


def train_model(model: MultiMLP, train_x: t.Tensor, train_y: t.Tensor, args):
    """Train all N models in parallel with shared data."""
    opt = t.optim.Adam(model.parameters(), lr=args.lr)
    loader = BatchedDataLoader(train_x, train_y, args.batch_size)

    for epoch in range(args.epochs):
        total_loss, n = 0.0, 0
        for bx, by in tqdm.tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            logits = model(bx)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), by.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n += 1
        print(f"  avg loss: {total_loss / n:.4f}")


@t.inference_mode()
def evaluate(model: MultiMLP, test_x: t.Tensor, test_y: t.Tensor) -> list[float]:
    """Per-model test accuracy.  Returns list of length n_models."""
    logits = model(test_x)
    return (logits.argmax(-1) == test_y).float().mean(1).tolist()


# ── Statistics ────────────────────────────────────────────────────────────────


def ci_95(arr) -> float:
    """95% confidence interval half-width of the mean."""
    if len(arr) < 2:
        return 0.0
    return 1.96 * float(np.std(arr)) / np.sqrt(len(arr))


def summarize(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "ci_95": ci_95(values), "n": len(values)}


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_bar(
    results: dict[str, list[float]],
    save_path: str,
    ylabel: str = "Test accuracy",
    title: str | None = None,
):
    """Bar chart with 95% CI error bars.

    Args:
        results: {condition_name: [per-run metric values]}
    """
    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    cis = [ci_95(v) for v in results.values()]

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.5), 4))
    x = range(len(names))
    ax.bar(x, means, yerr=cis, capsize=5, color="C0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=11)

    for i, (m, c) in enumerate(zip(means, cis)):
        ax.text(i, m + c + 0.003, f"{m:.1%}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")


def plot_sweep(
    sweep_results: dict[str, dict[str, list[float]]],
    save_path: str,
    xlabel: str = "Condition",
    ylabel: str = "Test accuracy",
    title: str | None = None,
):
    """Line plot with error bars for parameter sweeps.

    Args:
        sweep_results: {series_name: {x_label: [per-run values]}}
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_x_labels: list[str] = []
    for i, (series, data) in enumerate(sweep_results.items()):
        xs = list(data.keys())
        if not all_x_labels:
            all_x_labels = xs
        means = [np.mean(v) for v in data.values()]
        cis = [ci_95(v) for v in data.values()]
        ax.errorbar(
            range(len(xs)), means, yerr=cis, label=series,
            marker="o", capsize=4, linewidth=1.5, color=f"C{i}",
        )
    ax.set_xticks(range(len(all_x_labels)))
    ax.set_xticklabels(all_x_labels, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved {save_path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MLP with multi-run error bars")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_models", type=int, default=10,
                   help="Independent parallel runs for error bars")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256],
                   help="Hidden layer dimensions (input/output set automatically)")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--plot_dir", type=str, default="plots")
    return p.parse_args()


def make_tag(args) -> str:
    dims = "x".join(map(str, args.hidden_dims))
    return f"seed{args.seed}_n{args.n_models}_lr{args.lr:.0e}_ep{args.epochs}_h{dims}"


if __name__ == "__main__":
    args = parse_args()
    t.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds, test_ds, input_dim, num_classes = get_dataset()
    train_x_raw, train_y = to_tensors(train_ds)
    test_x_raw, test_y = to_tensors(test_ds)

    # expand() shares memory — data is stored only once on GPU
    train_x = train_x_raw.unsqueeze(0).expand(args.n_models, -1, -1, -1, -1)
    test_x = test_x_raw.unsqueeze(0).expand(args.n_models, -1, -1, -1, -1)

    sizes = [input_dim] + args.hidden_dims + [num_classes]
    print(f"Architecture: {sizes}, {args.n_models} parallel models")

    model = MultiMLP(args.n_models, sizes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) // args.n_models
    print(f"Params per model: {n_params:,}")

    # ── Evaluate untrained baseline (should be ~10% chance) ──
    baseline_accs = evaluate(model, test_x, test_y)

    # ── Train ──
    train_model(model, train_x, train_y, args)
    trained_accs = evaluate(model, test_x, test_y)

    baseline_summary = summarize(baseline_accs)
    trained_summary = summarize(trained_accs)
    print(f"\nBaseline:  {baseline_summary['mean']:.4f} ± {baseline_summary['ci_95']:.4f}")
    print(f"Trained:   {trained_summary['mean']:.4f} ± {trained_summary['ci_95']:.4f}")

    # ── Save results ──
    tag = make_tag(args)
    os.makedirs(args.results_dir, exist_ok=True)
    result = {
        "args": vars(args),
        "baseline": {"accuracies": baseline_accs, **baseline_summary},
        "trained": {"accuracies": trained_accs, **trained_summary},
    }
    json_path = os.path.join(args.results_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results → {json_path}")

    # ── Plot ──
    plot_bar(
        {"Random init": baseline_accs, "Trained": trained_accs},
        os.path.join(args.plot_dir, f"{tag}.pdf"),
        title="CIFAR-10 MLP (mean ± 95% CI)",
    )
