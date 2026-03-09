"""
JAX MLP Parallel Ablation Experiments
Efficiently runs learning rate and seed ablations in parallel using JAX vmap.

Usage:
    uv run jax_mlp_parallel.py --lr-values 1e-3 1e-4 1e-5 --seed-values 42 123 456
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap
import optax
import time
import numpy as np
from typing import Tuple, NamedTuple
from functools import partial

# Import optimized functions from jax_mlp_exp.py
from jax_mlp_exp import MLPParams, init_mlp_params, mlp_forward, compute_loss_and_grads


class ExperimentConfig(NamedTuple):
    """Configuration for a single experiment."""
    lr: float
    seed: int
    batch_size: int
    width: int
    wd: float
    epochs: int
    input_dim: int
    output_dim: int


@jit
def update_step(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray, 
                optimizer_state: optax.OptState, l2_lambda: float, 
                learning_rate: float) -> Tuple[MLPParams, optax.OptState, jnp.ndarray]:
    """Single training step."""
    loss, grads = compute_loss_and_grads(params, x, y, l2_lambda)
    updates, new_optimizer_state = optax.adam(learning_rate=learning_rate).update(
        grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_optimizer_state, loss


def train_epoch(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray, 
                optimizer_state: optax.OptState, batch_size: int, l2_lambda: float, 
                learning_rate: float) -> Tuple[MLPParams, optax.OptState, float]:
    """Train for one epoch."""
    n_samples = x.shape[0]
    n_batches = n_samples // batch_size
    
    @jit
    def body_fun(i, carry):
        params, optimizer_state, total_loss = carry
        start_idx = i * batch_size
        
        batch_x = jax.lax.dynamic_slice(x, (start_idx, 0), (batch_size, x.shape[1]))
        batch_y = jax.lax.dynamic_slice(y, (start_idx,), (batch_size,))
        
        params, optimizer_state, loss = update_step(params, batch_x, batch_y, 
                                                  optimizer_state, l2_lambda, learning_rate)
        return params, optimizer_state, total_loss + loss
    
    params, optimizer_state, total_loss = jax.lax.fori_loop(
        0, n_batches, body_fun, (params, optimizer_state, 0.0))
    
    return params, optimizer_state, total_loss / n_batches


@jit
def train_single_experiment(config_tuple: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Train a single experiment - vectorized for parallel execution."""
    lr, seed, batch_size, width, wd, epochs, input_dim, output_dim = config_tuple
    
    # Initialize with specific seed
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    params = init_mlp_params(subkey, int(input_dim), int(width), int(output_dim))
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=lr)
    optimizer_state = optimizer.init(params)
    
    # Training loop
    def scan_epoch(carry, epoch):
        params, optimizer_state, key = carry
        
        # Shuffle data
        key, subkey = random.split(key)
        perm = random.permutation(subkey, x.shape[0])
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        params, optimizer_state, avg_loss = train_epoch(params, x_shuffled, y_shuffled,
                                                      optimizer_state, int(batch_size), 
                                                      wd, lr)
        
        return (params, optimizer_state, key), avg_loss
    
    (params, _, _), losses = jax.lax.scan(scan_epoch, 
                                         (params, optimizer_state, key), 
                                         jnp.arange(int(epochs)))
    
    # Compute final accuracy
    logits = mlp_forward(params, x)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == y)
    
    return losses[-1], accuracy


def run_parallel_ablation(lr_values: list, seed_values: list, 
                         batch_size: int = 32, width: int = 512, 
                         wd: float = 1e-5, epochs: int = 100,
                         input_dim: int = 10, output_dim: int = 1000,
                         n_samples: int = 1000) -> dict:
    """Run parallel ablation experiments."""
    
    # Generate data once
    key = random.PRNGKey(0)
    key, subkey1, subkey2 = random.split(key, 3)
    train_x = random.normal(subkey1, (n_samples, input_dim))
    train_y = random.randint(subkey2, (n_samples,), 0, output_dim)
    
    # Create experiment configurations
    configs = []
    for lr in lr_values:
        for seed in seed_values:
            configs.append([lr, seed, batch_size, width, wd, epochs, input_dim, output_dim])
    
    config_array = jnp.array(configs)
    
    # Vectorized training function
    train_vectorized = vmap(train_single_experiment, in_axes=(0, None, None))
    
    # Run all experiments in parallel
    final_losses, accuracies = train_vectorized(config_array, train_x, train_y)
    
    # Organize results
    results = {
        'lr': jnp.array([c[0] for c in configs]),
        'seed': jnp.array([c[1] for c in configs]),
        'final_loss': final_losses,
        'accuracy': accuracies
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='JAX MLP parallel ablation experiments')
    parser.add_argument('--lr-values', nargs='+', type=float, default=[1e-3, 1e-4, 1e-5], 
                       help='Learning rate values to test')
    parser.add_argument('--seed-values', nargs='+', type=int, default=[42, 123, 456, 789, 999], 
                       help='Random seed values to test')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--width', type=int, default=512, help='Hidden layer width')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=1000, help='Output dimension')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of training samples')
    
    args = parser.parse_args()
    
    print(f"Using JAX device: {jax.devices()}")
    print(f"LR values: {args.lr_values}")
    print(f"Seed values: {args.seed_values}")
    print(f"Total experiments: {len(args.lr_values) * len(args.seed_values)}")
    
    start_time = time.time()
    
    # Run parallel experiments
    results = run_parallel_ablation(
        args.lr_values, args.seed_values, 
        args.batch_size, args.width, args.wd, args.epochs,
        args.input_dim, args.output_dim, args.n_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    print(f"{'LR':<10} {'Seed':<6} {'Final Loss':<12} {'Accuracy':<10}")
    print("-"*80)
    
    for i in range(len(results['lr'])):
        print(f"{results['lr'][i]:<10.1e} {results['seed'][i]:<6} {results['final_loss'][i]:<12.4f} {results['accuracy'][i]:<10.4f}")
    
    # Best experiment
    best_idx = jnp.argmin(results['final_loss'])
    print("\n" + "="*80)
    print("BEST EXPERIMENT")
    print("="*80)
    print(f"Learning Rate: {results['lr'][best_idx]:.1e}")
    print(f"Seed: {results['seed'][best_idx]}")
    print(f"Final Loss: {results['final_loss'][best_idx]:.4f}")
    print(f"Final Accuracy: {results['accuracy'][best_idx]:.4f}")
    
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
    print(f"Average time per experiment: {(time.time() - start_time) / len(results['lr']):.2f} seconds")


if __name__ == '__main__':
    main()