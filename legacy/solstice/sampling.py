# ---------------------------------------------------------------------------------------- #
#                                         UTILITIES                                        #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from scipy.stats.qmc import Halton
from tensorflow_probability.substrates.jax import distributions as tfd


# ------------------------------------ RANDOM SAMPLING ----------------------------------- #
def sample_mvg(key, N, d):
    m = tfd.MultivariateNormalDiag(jnp.zeros(d), jnp.ones(d))
    return m.sample(N, seed=key)

def sample_matern12(key, N, d):
    samples = jnp.random.uniform(key, (N, d))
    return jnp.tan(jnp.pi * (samples-0.5))

def sample_matern32(key, N, d):
    samples = jnp.random.uniform(key, (N, d))
    return (2*samples - 1) / jnp.sqrt(2* samples * (1-samples))

def sample_matern52(key, N, d):
    samples = jnp.random.uniform(key, (N, d))
    alpha = 4*samples*(1-samples)
    p = 4 * jnp.cos(jnp.arccos(jnp.sqrt(alpha))/3) / jnp.sqrt(alpha)
    return jnp.sign(samples-0.5) * jnp.sqrt(p-4)

# ------------------------------------- QMC SAMPLING ------------------------------------- #
def halton_samples(key: jax.random.PRNGKey, N: int, d: int):
    seed = key.tolist()[1]
    halton = Halton(d, scramble=True, seed=seed)
    samples = halton.random(N)
    return jnp.array(samples)

def qmc_mvg(key: jax.random.PRNGKey, N: int, d: int):
    samples = halton_samples(key, N, d)
    indep_normals = tfd.Normal(jnp.zeros(d), jnp.ones(d))
    return indep_normals.quantile(samples)

def qmc_matern12(key: jax.random.PRNGKey, N: int, d: int):
    samples = halton_samples(key, N, d)
    return jnp.tan(jnp.pi * (samples-0.5))

def qmc_matern32(key: jax.random.PRNGKey, N: int, d: int):
    samples = halton_samples(key, N, d)
    return (2*samples - 1) / jnp.sqrt(2* samples * (1-samples))

def qmc_matern52(key: jax.random.PRNGKey, N: int, d: int):
    samples = halton_samples(key, N, d)
    alpha = 4*samples*(1-samples)
    p = 4 * jnp.cos(jnp.arccos(jnp.sqrt(alpha))/3) / jnp.sqrt(alpha)
    return jnp.sign(samples-0.5) * jnp.sqrt(p-4)