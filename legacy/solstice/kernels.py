# ---------------------------------------------------------------------------------------- #
#                              EQUINOX KERNEL IMPLEMENTATIONS                              #
# ---------------------------------------------------------------------------------------- #
import jax
from jax import jit, vmap
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from .sampling import qmc_mvg, sample_mvg, halton_samples, sample_matern12,\
    sample_matern32, sample_matern52, qmc_matern12, qmc_matern32, qmc_matern52

# ------------------------------------------ RBF ----------------------------------------- #
class RBF(eqx.Module):
    variance: Float

    def __init__(self, variance = jnp.float32(1.)):
        self.variance = jnp.log(variance)

    def sample(self, key, shape, method="qmc"):
        if method == "qmc":
            return qmc_mvg(key, *shape)
        else:
            return sample_mvg(key, *shape)

    @jit
    def pairwise_distance(self, X):
        X1X1 = jnp.dot(X.T, X)
        x1_norm = jnp.diag(X1X1)
        dists = jnp.sqrt(jnp.maximum(x1_norm - 2 * X1X1 + x1_norm[:, None], 0))
        return dists

    @jit
    def evaluate(self, x1, x2):
        return jnp.exp(self.variance) * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2))

    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        # evaluate
        return vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)


# ---------------------------------------- MATERN ---------------------------------------- #
class Matern12(eqx.Module):  # also known as Laplace
    variance: Float

    def __init__(self, variance = jnp.float32(1.)):
        self.variance = jnp.log(variance)

    def sample(self, key, shape, method="qmc"):
        if method == "qmc":
            return qmc_matern12(key, *shape)
        else:
            return sample_matern12(key, *shape)

    @jit
    def evaluate(self, x1, x2):
        r = jnp.linalg.norm(x1 - x2)
        return jnp.exp(self.variance) * jnp.exp(-r)

    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        # evaluate
        return vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)


class Matern32(eqx.Module):  # also known as Laplace
    variance: Float

    def __init__(self, variance = jnp.float32(1.)):
        self.variance = jnp.log(variance)

    def sample(self, key, shape, method="qmc"):
        if method == "qmc":
            return qmc_matern12(key, *shape)
        else:
            return sample_matern12(key, *shape)

    @jit
    def evaluate(self, x1, x2):
        r = jnp.linalg.norm(x1 - x2)
        return jnp.exp(self.variance) * (1 + jnp.sqrt(3) * r) * jnp.exp(-jnp.sqrt(3) * r)

    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        # evaluate
        return vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)


class Matern52(eqx.Module):  # also known as Laplace
    variance: Float

    def __init__(self, variance = jnp.float32(1.)):
        self.variance = jnp.log(variance)

    def sample(self, key, shape, method="qmc"):
        if method == "qmc":
            return qmc_matern52(key, *shape)
        else:
            return sample_matern52(key, *shape)

    @jit
    def evaluate(self, x1, x2):
        r = jnp.linalg.norm(x1 - x2)
        kx1x2 = (1 + jnp.sqrt(5) * r + 5 * r ** 2 / 3) * jnp.exp(-jnp.sqrt(5) * r)
        return jnp.exp(self.variance) * kx1x2
    
    @jit
    def __call__(self, X1, X2):
        d_x1 = X1.shape[-1]
        d_x2 = X2.shape[-1]
        assert d_x1 == d_x2, "X1 and X2 must have the same number of dimensions"

        # make sure matrices
        X1 = X1.reshape(-1, d_x1)
        X2 = X2.reshape(-1, d_x2)

        # evaluate
        return vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)




# ------------------------------------------ RFF ----------------------------------------- #
class RFF(eqx.Module):
    w: Float[Array, "R d"]
    b: Float[Array, "R"]
    variance: Float

    def __init__(self, key, d, R, variance=jnp.float32(1.), base_kernel=None, sampling="qmc"):
        if base_kernel is None:
            base_kernel = RBF()

        self.w = base_kernel.sample(key, (R, d), method=sampling)
        # self.b = jax.random.uniform(key, (R,)) * 2 * jnp.pi
        self.b = (halton_samples(key, R, 1) * 2 * jnp.pi).reshape(-1)
        self.variance = jnp.log(variance)
    
    @property
    def R(self):
        return self.b.shape[0]

    @jit
    def phi(self, _X):
        cos_feats = jnp.sqrt(2) * jnp.cos(_X @ self.w.T + self.b)
        sin_feats = jnp.sqrt(2) * jnp.sin(_X @ self.w.T + self.b)
        projected = jnp.concatenate([cos_feats, sin_feats], axis=-1)

        return projected / jnp.sqrt(2 * self.R)

    @jit
    def evaluate(self, X1, X2):
        # altnerative RFF implementation cos(w^T(x1 - x2))
        cos_feats =  jnp.cos(self.w @ (X1 - X2))
        return cos_feats.mean()
    
    @jit 
    def evaluateK(self, X1, X2):
        K = vmap(vmap(self.evaluate, (None, 0)), (0, None))(X1, X2)
        return K

    @jit
    def A(self, X):
        phiX = self.phi(X)
        return jnp.exp(self.variance) * phiX.T @ phiX

    @jit
    def __call__(self, X1, X2):
        return jnp.exp(self.variance) * self.phi(X1) @ self.phi(X2).T
