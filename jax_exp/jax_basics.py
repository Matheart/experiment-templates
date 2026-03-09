import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp);

print(type(x_jnp))

x = jnp.arange(10)
#x[0] = 10
print(x)

y = x.at[0].set(2025)
print(y)

print(x.devices())
print(x.sharding) # can be across multiple devices

def norm(X):
    X -= X.mean(axis=0)
    return X / X.std(axis=0)

from jax import jit
norm_compiled = jit(norm)

np.random.seed(42)
X = jnp.array(np.random.randn(100, 10))
np.allclose(norm(X), norm_compiled(X), atol=1e-6)

from jax import grad

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

# jit can be combined with grad
print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

from jax import jacobian
print(jacobian(jnp.exp)(jnp.array([1.0, 2.0, 3.0])))

# vmap
from jax import random

key = random.key(1701)
key1, key2 = random.split(key)
mat = random.normal(key1, (150, 100))
batched_x = random.normal(key2, (10, 100)) # 10 batches

print(mat.shape)
print(batched_x.shape)

def apply_matrix(x):
  return jnp.dot(mat, x)

def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

@jit
def batched_apply_matrix(batched_x):
  return jnp.dot(batched_x, mat.T)

@jit
def vmap_batched_apply_matrix(batched_x):
  return jax.vmap(apply_matrix)(batched_x)

print('Naively batched')
a = naively_batched_apply_matrix(batched_x)
print('Manually batched') # These two much faster
b = batched_apply_matrix(batched_x)
print('Vmap batched')
c = vmap_batched_apply_matrix(batched_x)
np.testing.assert_allclose(a, b, atol=1e-4)
np.testing.assert_allclose(a, c, atol=1e-4)

# random key, never reuse key!
key = random.key(42)
print(key)

for i in range(3):
  new_key, subkey = random.split(key)
  del key

  val = random.normal(subkey)
  print(val)
  del subkey

  key = new_key

# array update
jax_array = jnp.zeros((3,3), dtype=jnp.float32)
updated_array = jax_array.at[1, :].set(1.0)
print("updated array:\n", updated_array)

print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = jax_array.at[::2, 3:].add(7.)
print("new array post-addition:")
print(new_jax_array)

