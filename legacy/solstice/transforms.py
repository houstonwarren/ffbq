# ---------------------------------------------------------------------------------------- #
#                                     INPUT TRANSFORMS                                     #
# ---------------------------------------------------------------------------------------- #

import jax
import jax.numpy as jnp
import equinox as eqx
from .kernels import RBF, RFF
from jax import jit, vmap
from jaxtyping import Array, Float
from jax.lax import cond

# ---------------------------------------- LINEAR ---------------------------------------- #
class ARD(eqx.Module):
    scale: Float[Array, "d"]
    kernel: eqx.Module

    def __init__(self, scale, base_kernel= RBF()):
        self.scale = jnp.log(scale)
        self.kernel = base_kernel

    # @jit
    # def phi(self, X):  # not sure how to do this better
    #     X = self.evaluate(X)
    #     return self.kernel.phi(X)

    @jit
    def evaluate(self, X):
        return X / jnp.exp(self.scale)  # constrained to positive values

    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = self.evaluate(X1.reshape(-1, d_x1))
        X2 = self.evaluate(X2.reshape(-1, d_x2))

        return self.kernel(X1, X2)


# ---------------------------------------- FINITE ---------------------------------------- #
class FiniteARD(eqx.Module):
    scale: Float[Array, "d"]
    kernel: eqx.Module
    support: Float[Array, "d"]
    # smoothing_scale: Float[Array, "d"]

    def __init__(self, scale, base_kernel, support):
        self.scale = jnp.log(scale)
        self.kernel = base_kernel

        if support.ndim > 1:
            support = jnp.abs(support[1, :] - support[0, :]) / 2

        self.support = support
        # self.smoothing_scale = jnp.log(support)

    @jit
    def cos_smoothing(self, dlr):
        term1_1 = (2 + jnp.cos(2 * jnp.pi * dlr)) / 3
        term1_2 = (1 - dlr)
        term2 = (1 / (2 * jnp.pi)) * jnp.sin(2 * jnp.pi * dlr)
        return term1_1 * term1_2 + term2

    @jit
    def boundary_fn_mahal(self, dx):
        # support_mod = jnp.minimum(jnp.exp(self.smoothing_scale), self.support)
        # support_mod = jnp.maximum(support_mod, self.support * 0.25)
        # r = jnp.sqrt(dx @ jnp.diag(1 / support_mod) @ dx.T)
        r = jnp.sqrt(dx @ jnp.diag(1 / self.support) @ dx.T)
        smoothing = jnp.where(
            r < 1, 
            self.cos_smoothing(r),
            0.
        )
        return smoothing

    @jit
    def boundary_fn_prod(self, dx):
        # support_mod = jnp.minimum(self.smth_scale, self.support)
        d_l = dx / self.support
        smoothing = jnp.where(
            (dx < self.support).all(), 
            self.cos_smoothing(d_l),
            0.
        )
        return jnp.prod(smoothing)

    @jit
    def evaluate(self, X):
        return X / jnp.exp(self.scale)  # constrained to positive values

    @jit
    def ard(self, X1, X2):  # run ARD only
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        return self.kernel(self.evaluate(X1), self.evaluate(X2))

    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        # apply smoothing
        K_smooth = vmap(vmap(
            lambda x1, x2: self.boundary_fn_mahal(jnp.abs(x1 - x2)), (None, 0)), (0, None)
        )(X1, X2)

        K_base = self.kernel(self.evaluate(X1), self.evaluate(X2))

        return K_smooth * K_base