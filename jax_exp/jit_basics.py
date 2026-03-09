import jax
import jax.numpy as jnp

global_list = []

def log2(x):
    print(x)
    global_list.append(x)
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0)) # JAX reduces each function into a sequence of primitive operations