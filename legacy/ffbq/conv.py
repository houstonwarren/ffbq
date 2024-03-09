# ---------------------------------------------------------------------------------------- #
#                                   CONVOLUTION OPERATORS                                  #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from tensorflow_probability.substrates.jax import distributions as tfd
from .measures import diag_mvgaussian_trunc_term, sample_truncated_measure, qmc, \
    measure_to_kernel_grid, scale_conv
from .helpers import area, grid, fft_interp, unpad, bounds_from_X
import scipy

# ---------------------------------- GAUSSIAN ANALYTICAL --------------------------------- #
def gaussian_conv(X, gp, m, bounds=None, R=10000, key=None):
    if bounds is None:
        return unbounded_gaussian_conv(X, gp.kernel, m)
    else:
        if key is None:
            raise ValueError("Must provide key for bounded convolution")
        return bounded_gaussian_conv(X, gp.kernel, m, bounds, R, key)


def bounded_gaussian_conv(X, k, m, bounds, R, key):
    z = bounded_gaussian_kmu(X, k, m, bounds)
    x_samples = qmc(key, m, R, bounds)
    # Z = bounded_gaussian_kmu(
    #     x_samples, k, m, bounds, normalize_k=False
    # ).mean().squeeze()
    Z = jax.vmap(lambda s: k(s, x_samples).mean())(x_samples).mean().squeeze() * area(bounds)**2
    # Z = k(x_samples, x_samples).mean().squeeze() * area(bounds)**2

    return z, Z


def bounded_gaussian_kmu(X, k, m, bounds, normalize_k=True):
    d = m.event_shape[0]
    Σ_ls = jnp.diag(jnp.exp(k.scale)**2)
    Σ_x = jnp.diag(m.stddev()**2)

    # posterior term of multiple of two gaussians
    C = jnp.linalg.inv(jnp.linalg.inv(Σ_ls) + jnp.linalg.inv(Σ_x))
    A_inv_a = (jnp.linalg.inv(Σ_ls) @ X.T).T
    B_inv_b = jnp.linalg.inv(Σ_x) @ m.mean()
    AB = A_inv_a + B_inv_b
    c = (C @ AB.T).T
    
    # normalizing constant
    normalizer = tfd.MultivariateNormalDiag(
        m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
    )
    norm_vals = normalizer.prob(X)

    # truncation terms
    posterior_truncs = vmap(
        lambda _c: diag_mvgaussian_trunc_term(_c, jnp.sqrt(jnp.diag(C)), bounds)
    )(c)
    p_x_trunc_term = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)

    # calculate output
    z = norm_vals * posterior_truncs
    z /= p_x_trunc_term
    z *= area(bounds)
    if normalize_k:
        z *= (2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(Σ_ls))
    return z


def unbounded_gaussian_conv(X, k, m):
    # # helpers
    d = m.event_shape[0]
    Σ_ls = jnp.diag(jnp.exp(k.scale)**2)
    Σ_x = jnp.diag(m.stddev()**2)

    # # normalizing constant
    normalizer = tfd.MultivariateNormalDiag(
        m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
    )
    norm_vals = normalizer.prob(X)
    # new_cov = jnp.linalg.inv(Σ_ls + Σ_x)
    # normalizer = jnp.exp(-0.5 * jax.vmap(lambda x: (x - m.mean()) @ new_cov @ (x - m.mean()))(X))
    # norm_vals = normalizer / (jnp.linalg.det(2 * jnp.linalg.inv(Σ_ls) @ Σ_x + jnp.eye(d)))**(1/2)

    # # calculate output
    z = norm_vals * ((2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(Σ_ls)))
        
    # # variance calculation
    Z = jnp.sqrt(jnp.exp(k.scale)**2 / (jnp.exp(k.scale)**2 + 2 * m.stddev()**2)).prod()
    # Z = jnp.sqrt(jnp.linalg.det(Σ_ls))
    # Z /=  jnp.sqrt(jnp.linalg.det(Σ_ls + 2 * Σ_x))
    return z, Z


# -------------------------------------- MONTE CARLO ------------------------------------- #
def mc_conv(X, gp, m, R, bounds=None, key=jax.random.PRNGKey(2023)):
    if bounds is None:
        return unbounded_mc_conv(X, gp.kernel, m, R=R, key=key)
    else:
        return bounded_mc_conv(X, gp.kernel, m, bounds, R=R, key=key)

def bounded_mc_conv(X, k, m, bounds, R, key):
    # sample points
    samples = sample_truncated_measure(m, R, bounds, key)
    var_samples = sample_truncated_measure(m, min(R, 10000), bounds, key)

    # run kernel matrix for mean and variance
    ab = area(bounds)
    # return samples, KXs
    z = jax.vmap(lambda x: k(x, samples).mean())(X) * ab
    Z = jax.vmap(lambda s: k(s, var_samples).mean())(var_samples).mean().squeeze() * area(bounds)**2
    # Z = k(var_samples, var_samples).mean() * ab**2
    return z, Z

def unbounded_mc_conv(X, k, m, R, key):
    samples = m.sample(R, key)
    var_samples = m.sample(min(R, 10000), key)

    z = jax.vmap(lambda x: k(x, samples).mean())(X)
    Z = k(var_samples, var_samples).mean()

    return z, Z

# ---------------------------------------- QMC ------------------------------------------- #
def qmc_conv(X, gp, m, R, bounds=None, key=jax.random.PRNGKey(2023)):
    if bounds is None:
        return unbounded_qmc_conv(X, gp.kernel, m, R=R, key=key)
    else:
        return bounded_qmc_conv(X, gp.kernel, m, bounds, R=R, key=key)


def unbounded_qmc_conv(X, k, m, R, key):
    samples = qmc(key, m , R)
    var_samples = qmc(key, m, min(R, 10000))

    z = jax.vmap(lambda x: k(x, samples).mean())(X)
    Z = k(var_samples, var_samples).mean()
    return z, Z

def bounded_qmc_conv(X, k, m, bounds, R, key):
    # sample points
    samples = qmc(key, m , R, bounds)
    var_samples = qmc(key, m, min(R, 10000), bounds)

    # run kernel matrix for mean and variance
    ab = area(bounds)
    # return samples, KXs
    z = jax.vmap(lambda x: k(x, samples).mean())(X) * ab
    Z = jax.vmap(lambda s: k(s, var_samples).mean())(var_samples).mean().squeeze() * area(bounds)**2
    # Z = k(var_samples, var_samples).mean() * ab**2

    return z, Z



# ------------------------------------------ RFF ----------------------------------------- #
def rff_conv(X, gp, m, R, bounds=None, key=jax.random.PRNGKey(2023)):
    if bounds is None:
        return unbounded_rff_conv(X, gp.kernel, m, R=R, key=key)
    else:
        return bounded_rff_conv(X, gp.kernel, m, bounds, R=R, key=key)


def unbounded_rff_conv(X, k, m, R, key):
    samples = qmc(key, m, R)
    X = X / jnp.exp(k.scale)
    samples = samples / jnp.exp(k.scale)

    phiX = k.kernel.phi(X)
    phiS = k.kernel.phi(samples).mean(axis=0)
    return phiX, phiS


def bounded_rff_conv(X, k, m, bounds, R, key):
    samples = qmc(key, m, R, bounds)
    X = X / jnp.exp(k.scale)
    samples = samples / jnp.exp(k.scale)
    
    ab = area(bounds)
    phiX = k.kernel.phi(X)
    phiS = k.kernel.phi(samples).mean(axis=0)
    return phiX, phiS


# ------------------------------------------ FFT ----------------------------------------- #
def fft_conv(X, gp, m, sr, bounds, bounded=True):
    N, d = X.shape

    # shift function if not centered about 0
    midpts = jnp.median(bounds, axis=0)
    if (midpts != 0).all():
        bounds = bounds - midpts
        X = X - midpts
        if isinstance(m, tfd.MultivariateNormalDiag):
            m = tfd.MultivariateNormalDiag(m.mean() - midpts, m.stddev())

    # make evaluation grid
    Xg = grid(bounds, sr, flatten=False)
    Xg_flat = Xg.reshape(-1, d)
    grid_dims = Xg.shape[:-1]

    # eval kernel and measure
    kX = gp.kernel(jnp.zeros((1, d)), Xg_flat).reshape(grid_dims)
    pX = m.prob(Xg_flat).reshape(grid_dims)
    trunc = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)
    if bounded:
        pX /= trunc

    ################# mean
    # convolve
    if d <= 2:
        zG_pad = jax.scipy.signal.fftconvolve(kX, pX, mode="full")
    else:
        zG_pad = jnp.array(scipy.signal.fftconvolve(kX, pX, mode="full"))
    
    # scale
    zG = unpad(zG_pad, grid_dims)
    # return zG, zG_pad
    zG_scaled = scale_conv(zG, bounds, inverse=False, bounded=bounded)
    # zG_scaled = zG
    if bounded:
        zG_scaled *= area(bounds)

    # interpolate
    z = fft_interp(X, zG_scaled, bounds, sr).reshape(-1)

    ################# variance
    # # shift grid for pX if needed
    # pX_shift = m.prob(grid(bounds, sr, centered=True)).reshape(grid_dims)
    # if bounded:
    #     shift_bounds = bounds_from_X(grid(bounds, sr, centered=True))
    #     trunc_shift = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), shift_bounds)
    #     pX_shift /= trunc_shift

    # convolve
    if d <= 2:
        # pad_shape = (jnp.asarray(grid_dims) * 2 -1).astype(int)
        # FkX = jnp.fft.fftshift(jnp.fft.fftn(kX, s=pad_shape))
        # FpX = jnp.fft.fftshift(jnp.fft.fftn(pX, s=pad_shape))
        # ZG = jnp.real(jnp.abs((jnp.fft.ifftn(FkX * FpX * FpX))))
        # # ZG = unpad(ZG, grid_dims)
        ZG = jax.scipy.signal.fftconvolve(zG, pX, mode="same")
    else:
        # pad_shape = (grid_dims * 2 -1).astype(int)
        # FkX = scipy.fft.fftn(kX, s=pad_shape)
        # FpX = scipy.fft.fftn(pX, s=pad_shape)
        # ZG = unpad(jnp.real(jax.scipy.signal.ifftn(FkX * FpX * FpX)), grid_dims)
        ZG = jnp.array(scipy.signal.fftconvolve(zG, pX, mode="same"))

    # scale
    # ZG = scale_conv(ZG, bounds, inverse=False, bounded=bounded)
    if bounded:
        # ZG = ZG * (area(bounds) / jnp.prod(jnp.asarray(grid_dims) - 1))**2 * area(bounds)**2
        ZG = ZG * (area(bounds) / jnp.prod(jnp.asarray(grid_dims) - 1))**2 * area(bounds)**2
    else:
        ZG  = ZG * (area(bounds) / jnp.prod(jnp.asarray(grid_dims) - 1))**2

    # interpolate
    Z = fft_interp(jnp.zeros((1, d)), ZG, bounds, sr).reshape(-1)

    return z, Z.squeeze()


# -------------------------------------- SPARSE FFBQ ------------------------------------- #
def sparse_fft_conv(X, gp, m, sr, bounds, bounded=True):
    N, d = X.shape

    # make evaluation grid
    Xg = grid(bounds, sr, flatten=False)
    Xg_flat = Xg.reshape(-1, d)
    grid_dims = Xg.shape[:-1]

    # eval kernel and measure
    phiG = gp.kernel.kernel.phi(gp.kernel.evaluate(Xg_flat))
    pX = m.prob(Xg_flat).reshape(grid_dims)
    trunc = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)
    if bounded:
        pX /= trunc

    # convolve and interpolate
    if d <= 2:
        phiG = jax.vmap(lambda x: x.reshape(grid_dims), 1)(phiG)
        zG = jax.vmap(lambda x: jax.scipy.signal.fftconvolve(x, pX, mode="same"), 0)(phiG)
        zG /= sr**d
        z = [fft_interp(jnp.zeros((1, d)), jnp.fft.fftshift(zg), bounds, sr) for zg in zG]
    else:
        z = []
        for col in range(phiG.shape[1]):
            phiG_r = phiG[:, col].reshape(grid_dims)
            zG = jnp.array(scipy.signal.fftconvolve(phiG_r, pX, mode="same"))
            zG /= sr**d
            _z = fft_interp(jnp.zeros((1, d)), jnp.fft.fftshift(zG), bounds, sr)
            z.append(_z)

    phiX = gp.kernel.kernel.phi(gp.kernel.evaluate(X))
    phiMu = jnp.stack(z).reshape(-1)

    return phiX, phiMu
