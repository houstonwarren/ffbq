# ---------------------------------------------------------------------------------------- #
#                                   PROBABILITY MEASURES                                   #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from scipy.stats.qmc import Halton
import jax.random as jr
from tensorflow_probability.substrates.jax import distributions as tfd
from scipy.stats.qmc import Halton
from solstice.transforms import ARD
from solstice.kernels import RBF
from solstice.sampling import halton_samples
from .helpers import grid, X_to_freq, scale_conv

# ----------------------------------------- UTILS ---------------------------------------- #
def sample_truncated_measure(m, N, bounds, key):
    d = m.event_shape[0]
    if isinstance(m, tfd.MultivariateNormalDiag):
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        samples = jax.random.uniform(key, (N, d), minval=lb_cdf, maxval=ub_cdf)
        samples = diag_mvgaussian_quantile(m, samples)
        return samples

    else:
        raise NotImplementedError("QMC sampling only implemented for multivariate normal")

    # # sample points
    # samples = m.sample(N, key)
    # lb, ub = bounds[0, :], bounds[1,:]
    # idx = jnp.where((samples >= lb).all(axis=1) & ((samples <= ub)).all(axis=1))[0]
    # samples = samples[idx, :]
    
    # # loop if not enough samples
    # sample_power = 0
    # while samples.shape[0] < N:
    #     sample_power += 1
    #     # if N * 2**sample_power * len(lb) > 1000000:
    #     #     return samples[0:N, :]
        
    #     samples = m.sample(N * 2**sample_power, key)
    #     idx = jnp.where((samples >= lb).all(axis=1) & ((samples <= ub)).all(axis=1))[0]
    #     samples = samples[idx, :]

    # samples = samples[0:N, :]
    # return samples


# ------------------------------------- QMC SAMPLING ------------------------------------- #
def qmc(key, m, N, bounds=None):
    if isinstance(m, tfd.MultivariateNormalDiag):
        return qmc_mvg(key, m, N, bounds=bounds)

    else:
        raise NotImplementedError("QMC sampling only implemented for multivariate normal")

# -------------------------------------- MV GAUSSIAN ------------------------------------- #
def diag_mvgaussian_trunc_term(mu, sigma, bounds):
    cdf_vals = jax.scipy.stats.norm.cdf(bounds, mu, sigma)
    cdf_diffs = jnp.abs(cdf_vals[1, :] - cdf_vals[0, :])
    return jnp.prod(cdf_diffs)


def diag_mvgaussian_quantile(m, X):
    # indep_normals = tfd.Normal(m.mean(), m.stddev())
    # return jax.vmap(indep_normals.quantile, 0)(X)
    return jax.scipy.stats.norm.ppf(X, m.mean(), m.stddev())


def qmc_mvg(key, m, N, bounds=None):
    d = m.event_shape[0]
    if bounds is None:
        samples = halton_samples(key, N, d)
        samples = diag_mvgaussian_quantile(m, samples)
    else:
        lb_cdf, ub_cdf = jax.scipy.stats.norm.cdf(bounds, m.mean(), m.stddev())
        diff = ub_cdf - lb_cdf
        samples = halton_samples(key, N, d)
        samples = lb_cdf + samples * diff
        samples = diag_mvgaussian_quantile(m, samples)
    return samples


# ----------------------------- MEASURE/KERNEL FOURIER DUALS ----------------------------- #
def measure_to_kernel_grid(m, bounds, sr, pad=True):
    # find apprpropriate kernel for measure
    if isinstance(m, tfd.MultivariateNormalDiag):
        kernel = ARD(m.stddev(), RBF())
    else:
        raise NotImplementedError("Only implemented for multivariate normal")
    
    trunc = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)

    # make evaluation grid
    d = bounds.shape[-1]
    if pad:
        Xg = grid(bounds * 2, sr, flatten=False)
    else:
        Xg = grid(bounds, sr, flatten=False)
    Xg_flat = Xg.reshape(-1, d)
    grid_dims = Xg.shape[:-1]

    # transform original inputs to frequencies
    freqs_flat = X_to_freq(Xg_flat, sr, shift=False)

    # evaluate the kernel at the frequencies
    Kf = kernel(jnp.zeros(d), freqs_flat).reshape(grid_dims)
    Kf /= trunc

    # scale to fft
    Kf = scale_conv(Xg_flat, Kf, inverse=True, single=True, pad=pad)
    return Kf
