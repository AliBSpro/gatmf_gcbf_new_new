from jaxtyping import Bool, Float

from jaxproxqp.utils.jax_types import Arr

XMat = Float[Arr, "nx nx"]
XVec = Float[Arr, "nx"]

EMat = Float[Arr, "ne nx"]
EVec = Float[Arr, "ne"]

IMat = Float[Arr, "ni nx"]
IVec = Float[Arr, "ni"]

IXVec = Float[Arr, "ni nx"]
IXBool = Bool[Arr, "ni nx"]

WVec = Float[Arr, "nw"]
