# ---------------------------------------------------------------------------------------- #
#                                     UTILITY FUNCTIONS                                    #
# ---------------------------------------------------------------------------------------- #

import jax
from jax import jit, vmap
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu


# -------------------------------------- PARAMETERS -------------------------------------- #
def frozen_partition(model, frozen_fn):
    # frozen fn looks like: lambda tree: (tree.b, tree.variance)
    filter_spec = jtu.tree_map(lambda _: eqx.is_array, model)
    filter_spec = eqx.tree_at(frozen_fn, filter_spec, replace_fn=lambda _: False)
    return eqx.partition(model, filter_spec)


def freeze(model, frozen_fn):
    filter_spec = jtu.tree_map(lambda _: eqx.is_array, model)
    filter_spec = eqx.tree_at(frozen_fn, filter_spec, replace_fn=lambda _: False)
    return filter_spec


def trainable(model, trainable_prms):
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(trainable_prms, filter_spec, replace_fn=lambda _: True)
    return eqx.partition(model, filter_spec)