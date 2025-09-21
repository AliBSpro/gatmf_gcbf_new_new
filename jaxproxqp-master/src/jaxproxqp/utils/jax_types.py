from __future__ import annotations
from typing import Union, Tuple
import warnings

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

Arr = Union[np.ndarray, Array]

AnyFloat = Float[Arr, "*"]
Shape = Tuple[int, ...]

FloatScalar = Union[float, Float[Arr, ""]]
IntScalar = Union[int, Int[Arr, ""]]
BoolScalar = Bool[Arr, ""]


def get_default_float_dtype():
    """Get the default float dtype for JAX."""
    return jnp.zeros(0).dtype


def get_default_int_dtype():
    """Get the default int dtype for JAX."""
    return jnp.zeros(0, dtype=int).dtype


def get_default_bool_dtype():
    """Get the default bool dtype for JAX."""
    return jnp.zeros(0, dtype=bool).dtype


def assert_x64_enabled():
    """Assert that JAX is configured to use 64-bit precision."""
    if not jnp.zeros(1).dtype == jnp.float64:
        warnings.warn("JAX is not configured for 64-bit precision. Results may be inaccurate.")


def float32_is_default():
    """Check if JAX is using float32 as default."""
    return jnp.zeros(1).dtype == jnp.float32