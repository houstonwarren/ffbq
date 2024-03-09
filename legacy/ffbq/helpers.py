# ---------------------------------------------------------------------------------------- #
#                                     UTILITY FUNCTIONS                                    #
# ---------------------------------------------------------------------------------------- #
import jax
import jax.numpy as jnp
from jax import vmap, jit
from scipy.interpolate import interpn

# ----------------------------------- KERNEL UTILITIES ----------------------------------- #
@jit
def pairwise_distance(X):
    X1X1 = jnp.dot(X.T, X)
    x1_norm = jnp.diag(X1X1)
    dists = jnp.sqrt(jnp.maximum(x1_norm - 2 * X1X1 + x1_norm[:, None], 0))
    return dists


def stabilize(K, alpha=None):
    # Perform eigenvalue decomposition
    eigvals, eigvecs = jnp.linalg.eigh(K)
    
    if alpha is None:
        alpha = eigvals.max() * 0.001

    # Adjust eigenvalues below the threshold alpha
    adjusted_eigvals = jnp.maximum(eigvals, alpha)
    
    # Reconstruct the covariance matrix
    K_adjusted = eigvecs @ jnp.diag(adjusted_eigvals) @ eigvecs.T
    
    return K_adjusted, alpha

# ------------------------------------- FFT UTILITIES ------------------------------------ #
def area(bounds):
    return jnp.prod(jnp.abs(bounds[1, :] - bounds[0, :]))


def shift_axis(ax):
    shift_val = (ax[1:] - ax[:-1]).mean()
    return ax - shift_val / 2


def grid(bounds, sampling_rate, flatten=True, axes_only=False, centered=False):
    dim_N = jnp.abs(bounds[1, :] - bounds[0,:]) * sampling_rate

    axes = [
        jnp.linspace(bounds[0, i], bounds[1, i], int(dim_N[i]))
        for i in range(dim_N.shape[0])
    ]

    dimN = jnp.array([len(ax) for ax in axes])
    if centered:
        axes = [
            shift_axis(axes[i]) if dimN[i] % 2 == 0 else axes[i] 
            for i in range(len(axes))
        ]

    if axes_only:
        return axes

    grid = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1)
    if flatten:
        grid = grid.reshape(-1, grid.shape[-1])

    return grid


def regrid(data, d):  # assumes that the resulting grid is hypercubic
    assert data.ndim == 2 or data.ndim == 1
    
    N = data.shape[0]
    Nd = round(N**(1/d))
    axes_Nd = tuple([Nd for _ in range(d)])

    if data.ndim == 1:
        data = data.reshape(*axes_Nd)
    else:  # given assert above this means data.ndim == 2
        if d == 1:
            data = data.reshape(*axes_Nd)
        else:
            data = data.reshape(*axes_Nd, d)
    return data


def smart_grid_inputs(X):  # assumes that the bounds are the same for all dims
    """Process whether a grid is needed or not for multi-d inputs.
    Regrid if necessary.

    Args:
        X (jnp.ndarray): (n * d) grid or flat points

    Returns:
        jnp.ndarray: n1 x n2 x ... x nd x d grid of points
    """

    if X.ndim == 1:
        raise ValueError("X must be at least 2D")

    elif X.ndim > 1:
        # if properly gridded then len(X.shape[:-1]) == X.shape[-1]
        if len(X.shape[:-1]) == X.shape[-1]:
            X_grid = X

        else:
            X_grid = regrid(X, X.shape[-1])

    return X_grid


def fftfreqn(dims: tuple, sr, shift=True, flatten=True):
    # get frequencies for each dim
    freqs = [
        jnp.fft.fftfreq(n_dim, 1 / sr) for n_dim in dims
    ]
    if shift:
        freqs = [jnp.fft.fftshift(freq) for freq in freqs]

    # get meshgrid
    grid = jnp.stack(jnp.meshgrid(*freqs, indexing='ij'), axis=-1)
    if flatten:
        grid = grid.reshape(-1, grid.shape[-1])

    return grid


def X_to_freq(X, sr, shift=True, flatten=True):
    X_grid = smart_grid_inputs(X)
    
    # get frequencies
    grid_dims = X_grid.shape[:-1]
    freqs = fftfreqn(grid_dims, sr, shift=shift, flatten=flatten)
    freqs = freqs
    return freqs


def bounds_from_X(X):
    # get bounds
    bounds = jnp.stack([
        jnp.amin(X, axis=tuple(range(X.ndim - 1))),
        jnp.amax(X, axis=tuple(range(X.ndim - 1)))
    ], axis=0)

    return bounds


def scale_conv(conv, bounds, inverse=False, bounded=True):
    """Scale convolution to be consistent with the original data

    Args:
        X_orig (jnp.ndarray): Original data
        conv (jnp.ndarray): Convolution grid
        inverse (bool, optional): whether to scale original data to convolultion size. 
            Defaults to False.
        single (bool, optional): Whether conv is a convolution or single FFT. 
            Defaults to False.

    Returns:
        jnp.ndarray: _description_
    """

    area_X = area(bounds)
    dimsN = jnp.array(conv.shape) - 1
    d = dimsN.shape[0]

    # create scaling factor
    scaling_factor = area_X / jnp.prod(dimsN)

    if inverse:
        scaling_factor = 1 / scaling_factor

    # scale convolution
    conv_scaled = conv * scaling_factor

    return conv_scaled


def fft_interp(X, conv, bounds, sr, center=True):
    Xg_ax = grid(bounds, sr, axes_only=True, centered=center)

    X_interp = interpn(
        Xg_ax, conv, X, 
        method="pchip", bounds_error=False, fill_value=None
    )
    return X_interp


def fftpad(gridY):
    """Pad a matrix with trailing (N_d - 1, ...) zeros 

    Args:
        Y (jnp.ndarray): Uniformly spaced gridded matrix to pad
        leading (tuple): Leading dimensions to pad
        trailing (tuple): Trailing dimensions to pad

    Returns:
        jnp.ndarray: Padded matrix
    """
    # gridY = smart_grid(Y)
    
    return jnp.pad(gridY, [(0, nd - 1) for nd in gridY.shape])


def unpad(Xpad, newshape):
    currshape = jnp.asarray(Xpad.shape)
    newshape = jnp.asarray(newshape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = tuple(slice(startind[i], endind[i]) for i in range(len(startind)))
    return Xpad[myslice]
