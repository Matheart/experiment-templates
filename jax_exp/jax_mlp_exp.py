"""
JAX MLP training for experimentation.
A simple two-layer multilayer perceptron with configurable hyperparameters using JAX.

lr sweep: 1e-3, 1e-4, 1e-5
wd sweep: 1e-3, 1e-4, 1e-5, 1e-6
batch size sweep: 32, 64, 128, 256, 512
width sweep: 64, 128, 256, 512, 1024, 2048
Total: 3 * 4 * 5 * 6 = 360 experiments

uv run jax_mlp_exp.py --lr 1e-3 --batch-size 32 --wd 1e-5 --width 512
uv run jax_mlp_exp.py --lr 1e-4 --batch-size 32 --wd 1e-5 --width 512 
uv run jax_mlp_exp.py --lr 1e-5 --batch-size 32 --wd 1e-5 --width 512 
"""

import argparse
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, grad, vmap
import optax
import time
import numpy as np
from typing import Tuple, NamedTuple


class MLPParams(NamedTuple):
    """Parameters for the MLP model."""
    w1: jnp.ndarray  # First layer weights
    b1: jnp.ndarray  # First layer bias
    w2: jnp.ndarray  # Second layer weights
    b2: jnp.ndarray  # Second layer bias


def init_mlp_params(key: random.PRNGKey, input_dim: int, width: int, output_dim: int) -> MLPParams:
    """Initialize MLP parameters."""
    key1, key2 = random.split(key)
    
    # Xavier initialization
    w1 = random.normal(key1, (input_dim, width)) * jnp.sqrt(2.0 / input_dim)
    b1 = jnp.zeros(width)
    w2 = random.normal(key2, (width, output_dim)) * jnp.sqrt(2.0 / width)
    b2 = jnp.zeros(output_dim)
    
    return MLPParams(w1, b1, w2, b2)


@jit
def mlp_forward(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through the MLP."""
    # First layer: Linear + ReLU
    h = jnp.dot(x, params.w1) + params.b1
    h = jnp.maximum(0, h)  # ReLU activation
    
    # Second layer: Linear
    logits = jnp.dot(h, params.w2) + params.b2
    return logits


@jit
def compute_loss_and_grads(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray, l2_lambda: float) -> Tuple[jnp.ndarray, MLPParams]:
    """Compute loss and gradients in one pass for efficiency."""
    def total_loss(p):
        logits = mlp_forward(p, x)
        
        # Cross-entropy loss with numerical stability
        log_probs = jax.nn.log_softmax(logits)
        ce_loss = -jnp.mean(log_probs[jnp.arange(len(y)), y])
        
        # L2 regularization
        reg_loss = l2_lambda * (jnp.sum(p.w1 ** 2) + jnp.sum(p.w2 ** 2))
        return ce_loss + reg_loss
    
    loss, grads = jax.value_and_grad(total_loss)(params)
    return loss, grads


@jit
def update_step(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray, 
                optimizer_state: optax.OptState, l2_lambda: float, 
                learning_rate: float) -> Tuple[MLPParams, optax.OptState, jnp.ndarray]:
    """Single training step."""
    loss, grads = compute_loss_and_grads(params, x, y, l2_lambda)
    
    # Create optimizer update function directly to avoid tracing issues
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
        
        # Use dynamic_slice for JAX compatibility
        batch_x = jax.lax.dynamic_slice(x, (start_idx, 0), (batch_size, x.shape[1]))
        batch_y = jax.lax.dynamic_slice(y, (start_idx,), (batch_size,))
        
        params, optimizer_state, loss = update_step(params, batch_x, batch_y, 
                                                  optimizer_state, l2_lambda, learning_rate)
        return params, optimizer_state, total_loss + loss
    
    # Use fori_loop for better performance than scan
    params, optimizer_state, total_loss = jax.lax.fori_loop(
        0, n_batches, body_fun, (params, optimizer_state, 0.0))
    
    return params, optimizer_state, total_loss / n_batches


def train(params: MLPParams, x: jnp.ndarray, y: jnp.ndarray, 
          learning_rate: float, epochs: int, batch_size: int, 
          l2_lambda: float, key: random.PRNGKey) -> Tuple[MLPParams, jnp.ndarray]:
    """Train the model for multiple epochs."""
    # Initialize optimizer state once
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)
    
    def scan_epoch(carry, epoch):
        params, optimizer_state, key = carry
        
        # Efficient shuffling using indices instead of copying data
        key, subkey = random.split(key)
        perm = random.permutation(subkey, x.shape[0])
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        params, optimizer_state, avg_loss = train_epoch(params, x_shuffled, y_shuffled,
                                                      optimizer_state, batch_size, 
                                                      l2_lambda, learning_rate)
        
        return (params, optimizer_state, key), avg_loss
    
    (params, _, _), losses = jax.lax.scan(scan_epoch, 
                                         (params, optimizer_state, key), 
                                         jnp.arange(epochs))
    
    return params, losses


def main():
    parser = argparse.ArgumentParser(description='JAX MLP training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--width', type=int, default=512, help='Hidden layer width')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=5000, help='Output dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    key = random.PRNGKey(args.seed)
    print(f"Using JAX device: {jax.devices()}, seed: {args.seed}")

    start_time = time.time()
    
    # Generate dummy data for demonstration
    # In practice, replace this with your actual dataset
    key, subkey1, subkey2 = random.split(key, 3)
    train_x = random.normal(subkey1, (1000, args.input_dim))
    train_y = random.randint(subkey2, (1000,), 0, args.output_dim)
    
    # Initialize model parameters
    key, subkey = random.split(key)
    params = init_mlp_params(subkey, args.input_dim, args.width, args.output_dim)

    print(f"Starting training with lr={args.lr}, batch_size={args.batch_size}, "
          f"wd={args.wd}, width={args.width}")
    
    # Train
    trained_params, losses = train(params, train_x, train_y, args.lr, args.epochs, 
                                  args.batch_size, args.wd, key)
    
    # Print training progress every 10 epochs
    for i in range(0, args.epochs, 10):
        jax.debug.print("Epoch [{}/{}], Loss: {:.4f}", i+1, args.epochs, losses[i])
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
