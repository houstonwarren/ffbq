# ---------------------------------------------------------------------------------------- #
#                                EQUINOX GP IMPLEMENTATIONS                                #
# ---------------------------------------------------------------------------------------- #

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, Bool
from .kernels import RFF
from .transforms import ARD
from jax import jit, vmap
from functools import partial
from jax.lax import cond
from .utils import frozen_partition, trainable
import optax
from progressbar import progressbar

# --------------------------------------- BASIC GP --------------------------------------- #
class GP(eqx.Module):
    kernel: eqx.Module
    X: Float[Array, "N d"]
    diag: Float
    
    def __init__(self, kernel, X, diag=None):
        
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = diag

    @partial(jax.jit, static_argnums=(2,))
    def nll(self, y, solver="full"):
        return cond(solver == "chol", self.chol_nll, self.full_nll, y)

    @jit
    def full_nll(self, y):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self.diag
        n = y.shape[0]
        term1 = 0.5 * jnp.dot(y.T, jnp.linalg.solve(K, y))
        term2 = 0.5 * jnp.linalg.slogdet(K)[1]
        term3 = 0.5 * n * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

    @jit
    def chol_nll(self, y):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self.diag
        n = y.shape[0]
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        term1 = 0.5 * jnp.dot(y.T, alpha)
        term2 = jnp.sum(jnp.log(jnp.diag(L)))
        term3 = 0.5 * n * jnp.log(2 * jnp.pi)
        return term1 + term2 + term3

    @jit
    def condition(self, y, X_test=None):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self.diag
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test) + jnp.eye(X_test.shape[0]) * self.diag
        
        K_inv = jnp.linalg.inv(K)
        
        mu_pred = jnp.dot(K_star.T, jnp.linalg.solve(K, y))
        sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(K_star.T, jnp.dot(K_inv, K_star))))
        
        return jnp.array([mu_pred, sigma_pred])

    @jit
    def chol_condition(self, y, X_test):
        K = self.kernel(self.X, self.X) + jnp.eye(self.X.shape[0]) * self.diag
        K_star = self.kernel(self.X, X_test)
        K_star_star = self.kernel(X_test, X_test) + jnp.eye(X_test.shape[0]) * self.diag
        
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        
        mu_pred = jnp.dot(K_star.T, alpha)
        v = jnp.linalg.solve(L, K_star)
        sigma_pred = jnp.sqrt(jnp.diag(K_star_star - jnp.dot(v.T, v)))
        
        return jnp.array([mu_pred, sigma_pred])

    @partial(jax.jit, static_argnums=(3,))
    def __call__(self, y, X_test, solver="full"):
        mu, sigma = cond(solver == "chol", self.condition, self.chol_condition, y, X_test)

        # mu, sigma = jnp.where(
        #     solver == "chol", self.chol_condition(y, X_test), self.condition(y, X_test)
        # )
        return mu, sigma


# -------------------------------------- LOW RANK GP ------------------------------------- #
class LowRankGP(eqx.Module):
    kernel: ARD
    X: Float[Array, "N d"]
    diag: Float

    def __init__(self, kernel, X, diag=None):
        self.X = X
        self.kernel = kernel

        if diag is None:
            diag = jnp.float32(1e-5)
        self.diag = diag

    @partial(jax.jit, static_argnums=(2,))
    def nll(self, y, solver="chol"):
        return cond(solver == "chol", self.chol_nll, self.chol_nll, y)

    @jit
    def chol_nll(self, y):
        diag = self.diag
        Xtransform = self.kernel.evaluate(self.X)
        phiX = self.kernel.kernel.phi(Xtransform)
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * diag
        R = jnp.linalg.cholesky(A)
        y_loss = jnp.linalg.solve(R, phiX.T @ y)

        lml_1 = -((y.T @ y) - (y_loss.T @ y_loss)) / (2 * diag)
        lml_2 = -0.5 * jnp.sum(jnp.log(jnp.diag(R)**2))
        lml_3 = m * jnp.log(m * diag)
        lml_4 = -0.5 * n * jnp.log(2 * jnp.pi * diag)
        lml = lml_1 + lml_2 + lml_3 + lml_4
        return -lml

    @jit
    def condition(self, y_train, X_test, diag):
        phiXt = self.kernel.kernel.phi(self.kernel.evaluate(X_test))
        phiX = self.kernel.kernel.phi(self.kernel.evaluate(self.X))
        n, m = phiX.shape
        A = phiX.T @ phiX + jnp.eye(m) * diag
        R = jnp.linalg.cholesky(A)

        # mean calculation
        R_phiy = jnp.linalg.solve(R, phiX.T @ y_train)
        y_pred = jnp.linalg.solve(R.T, R_phiy)
        mu = phiXt @ y_pred

        # variance calculation
        R_phixt = jnp.linalg.solve(R, phiXt.T)
        V = phiXt @ R_phixt * diag
        # V = noise_std**2 + (R_phixt @ R_phixt.T * noise_std**2)
        sigma = jnp.sqrt(jnp.diag(V))

        return mu, sigma

    @partial(jax.jit, static_argnums=(4,))
    def __call__(self, y_train, X_test, diag=jnp.float32(0.), solver="chol"):
        diag = cond(diag == 0., lambda: self.diag, lambda: diag)
        return self.condition(y_train, X_test, diag)


# --------------------------------------- TRAINING --------------------------------------- #
def fitgp(gp, y, opt, epochs, to_train=None, 
        frozen=lambda tree: (tree.X, tree.diag), 
        **kwargs):

    if to_train is not None:
        params, static = trainable(gp, to_train)
    else:
        params, static = frozen_partition(gp, frozen)

    # construct loss function
    solver = kwargs.get("solver", "chol")

    # define an opt step
    @jit
    def opt_step(params, opt_state, y):
        @jax.value_and_grad
        def loss_fn(params, static, y):
            model = eqx.combine(params, static)
            return model.nll(y, solver=solver)

        loss, grads = loss_fn(params, static, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # initalize optimizer
    opt_state = opt.init(gp)

    # loop over epochs
    verbose = kwargs.get("verbose", False)
    print_iter = kwargs.get("print_iter", 25)
    loss_vals = []
    if verbose:
        iterator = progressbar(range(epochs), redirect_stdout=True)
    else:
        iterator = range(epochs)
    
    loss_vals = []
    for epoch in iterator: 
        params, opt_state, loss = opt_step(params, opt_state, y)
        loss_vals.append([epoch, loss])
        # # print output
        if verbose and epoch % print_iter == 0:
            print(f"epoch {epoch},loss: {loss}")
    
    model = eqx.combine(params, static)
    return model, jnp.array(loss_vals)