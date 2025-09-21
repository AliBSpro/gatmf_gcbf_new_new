import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from jaxproxqp.utils.jax_types import Arr, FloatScalar


def infty_norm(x, axis: int = None):
    if x.size == 0:
        if axis is None:
            return 0.0

        shape = list(x.shape)
        del shape[axis]
        return np.zeros(shape)
    return jnp.abs(x).max(axis)


def sqnorm(x):
    if x.size == 0:
        return 0.0
    return jnp.sum(x**2)


def pos_part(x):
    return jnp.where(x > 0, x, 0)


def neg_part(x):
    return jnp.where(x < 0, x, 0)


def add_diag_seg(mat: Float[Arr, "n n"], start_idx: int, fill_len: int, value: FloatScalar) -> Float[Arr, "n n"]:
    iota = start_idx + jnp.arange(fill_len)
    return mat.at[iota, iota].add(value)


def default_float():
    return jnp.ones(0).dtype


def get_machine_eps():
    return np.finfo(default_float()).eps
