# ---------------------------------------------------------------------------------------- #
#                                    BAYESIAN QUADRATURE                                   #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.sparse.linalg import cg
from .conv import gaussian_conv, mc_conv, qmc_conv, fft_conv, rff_conv, sparse_fft_conv
from .helpers import area, stabilize

# --------------------------------------- BASIC BQ --------------------------------------- #
class BQ:
    def __init__(self, gp, m, operator="gaussian", key=jax.random.PRNGKey(2023)):
        self.gp = gp
        self.m = m
        self.key = key
        
        if operator == "gaussian":
            self.operator = gaussian_conv

        if operator == "qmc":
            self.operator = qmc_conv

        if operator == "mc":
            self.operator = mc_conv

        if operator == "fft":
            self.operator = fft_conv

        if operator == "lr":
            self.operator = rff_conv

        if operator == "lrfft":  # sparse fft
            self.operator = sparse_fft_conv

    def kmu_kvar(self, X, bounds=None, **kwargs):
        if self.operator != rff_conv and self.operator != sparse_fft_conv:
            Zs = self.operator(X, self.gp, self.m, bounds=bounds, **kwargs)
        elif self.operator == sparse_fft_conv:
            phiX, phiS = sparse_fft_conv(X, self.gp, self.m, bounds=bounds, **kwargs)
            z = phiX @ phiS.T
            Z = phiS.T @ phiS
            Zs = z, Z
        elif self.operator == rff_conv:
            phiX, phiS = rff_conv(X, self.gp, self.m, bounds=bounds, **kwargs)
            z = phiX @ phiS.T
            Z = phiS.T @ phiS
            Zs = z, Z
        return Zs
    
    def zK_z(self, X, z, diag=None):
        N_x, d = X.shape
        K = self.gp.kernel(X, X)

        if diag is not None:
            K = K + jnp.eye(N_x) * diag
            K = jnp.clip(K, 0, None)
            # K, diag_effective = stabilize(K, alpha=diag)
        else:
            if self.operator != gaussian_conv:
                K, diag_effective = stabilize(K, diag)
                K = jnp.clip(K, 0, None)
            else:
                K = K + jnp.eye(N_x) * jnp.trace(K) * 0.001
                # K, diag_effective = stabilize(K, diag)
                K = jnp.clip(K, 0, None)

        # solve through conjugate gradients
        zK = cg(K, z)[0]
        zKz = z @ zK
        # zKz *= 1 / (1 + diag)

        return zK, zKz

    def __call__(self, X, y, diag=None, bounds=None, areaX=1, **kwargs):
        if self.operator != rff_conv and self.operator != sparse_fft_conv:

            # kernel mean and variance
            z, Z = self.kmu_kvar(X, bounds=bounds, **kwargs)
            zK, zKz = self.zK_z(X, z, diag=diag)

            # posterior mu and sigma of integral estimate
            # K_inv = jnp.linalg.inv(K)
            # L = jnp.linalg.cholesky(K)
            # alpha = jax.scipy.linalg.solve_triangular(L, y, lower=True)
            # beta = jax.scipy.linalg.solve_triangular(L, z, lower=True)
            # mu = jnp.dot(alpha, beta).reshape(-1)

            mu = (y @ zK).squeeze()
            
            # variance
            variance = (Z - zKz).squeeze() 
            if bounds is not None:
                variance /= area(bounds)**2

            return jnp.array([mu, variance]).reshape(-1) 
        
        else:
            if self.operator == sparse_fft_conv:
                phiX, phiS = sparse_fft_conv(X, self.gp, self.m, bounds=bounds, **kwargs)
            else:
                phiX, phiS = rff_conv(X, self.gp, self.m, bounds=bounds, **kwargs)
            
            R = phiX.shape[1]
            N = phiX.shape[0]

            areaX = 1
            if bounds is not None:
                areaX = area(bounds)
            phiS *= areaX
            
            A = phiX.T @ phiX
            diag = jnp.trace(A) * 0.0001
            if jnp.isnan(diag) or diag < 0:
                diag = 1e-3
            A = A + jnp.eye(R) * diag
            # A = stabilize(A, alpha=diag)[0]
            Akmu = cg(A, phiS)[0]

            # mean
            mu = (y @ phiX @ Akmu).squeeze()

            # variance
            variance = (phiS @ Akmu / areaX**2).squeeze()

            # # variance - full rank
            # z = phiX @ phiS.T # * areaX
            # Z = phiS.T @ phiS #* areaX**2
            # _, zKz = self.zK_z(X, z, diag=diag)
            # variance = (Z - zKz).squeeze() / areaX**4

            return jnp.stack([mu, variance]).reshape(-1)



        




# -------------------------------------- LOW RANK BQ ------------------------------------- #