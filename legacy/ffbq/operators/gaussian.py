# ---------------------------------------------------------------------------------------- #
#                                 RBF / GAUSSIAN MEASURE BQ                                #
# ---------------------------------------------------------------------------------------- #

import jax
import jax.numpy as jnp
from jax import vmap, jit
from tensorflow_probability.substrates.jax import distributions as tfd
from ..measures import diag_mvgaussian_trunc_term, sample_truncated_measure, qmc, \
    measure_to_kernel_grid, scale_conv
from ..helpers import area, grid, fft_interp, unpad, bounds_from_X
import scipy
import equinox as eqx


class GaussianConv(eqx.Module):
    measure: tfd.MultivariateNormalDiag

    def __init__(self, measure):
        self.measure = measure

    def pX_pXp(self, X, gp):
        """Apply Gaussian identities to calculate p(x) * p(x').

        Args:
            X (jnp.ndarray): Data points.
            gp (solstice.gps.GP): Gaussian process.
        """
        m = self.measure
        k = gp.kernel
        d = self.measure.event_shape[0]
    
        Σ_ls = jnp.diag(jnp.exp(k.scale)**2)
        Σ_x = jnp.diag(m.stddev()**2)

        # first distribution N(X | m.mean(), Σ_ls + Σ_x)
        normalizer = tfd.MultivariateNormalDiag(
            m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
        )

        # second distribution N(x | C(Σ_ls X + Σ_x m.mean()), C)
        C = jnp.linalg.inv(jnp.linalg.inv(Σ_ls) + jnp.linalg.inv(Σ_x))

        # posterior term of multiple of two gaussians
        
        A_inv_a = (jnp.linalg.inv(Σ_ls) @ X.T).T
        B_inv_b = jnp.linalg.inv(Σ_x) @ m.mean()
        AB = A_inv_a + B_inv_b
        c = (C @ AB.T).T
        
        # normalizing constant

        norm_vals = normalizer.prob(X)

        # truncation terms
        posterior_truncs = vmap(
            lambda _c: diag_mvgaussian_trunc_term(_c, jnp.sqrt(jnp.diag(C)), bounds)
        )(c)
        p_x_trunc_term = diag_mvgaussian_trunc_term(m.mean(), m.stddev(), bounds)

        


    def z(self, X, gp):
        kernel = gp.kernel
        m = self.measure

        d = m.event_shape[0]
        Σ_ls = jnp.diag(jnp.exp(kernel.scale)**2)
        Σ_x = jnp.diag(m.stddev()**2)

        # # normalizing constant
        normalizer = tfd.MultivariateNormalDiag(
            m.mean(), jnp.sqrt(jnp.diag(Σ_ls + Σ_x))
        )
        norm_vals = normalizer.prob(X)

        # # calculate output
        z = norm_vals * ((2 * jnp.pi)**(d/2) * jnp.sqrt(jnp.linalg.det(Σ_ls)))
        return z

    def bounded_z(self, X, gp, bounds, R, key):
        pass

    def Z(self, X, gp):
        kernel = gp.kernel
        m = self.measure
        Z = jnp.sqrt(jnp.exp(kernel.scale)**2 / (jnp.exp(kernel.scale)**2 + 2 * m.stddev()**2)).prod()
        return Z

    def bounded_Z(self, X, gp, bounds, R, key):
        pass

    def __call__(self, X, gp, bounds=None, R=10000, key=None):
        if bounds is None:
            return self.z(X, gp)
        else:
            if key is None:
                raise ValueError("Must provide key for bounded convolution")
            return self.bounded_z(X, gp, bounds, R, key)
    

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
