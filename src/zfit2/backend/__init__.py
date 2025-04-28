"""Backend module for scientific computing.

This module provides a unified interface to different computational backends:
- JAX: Accelerated array computing and automatic differentiation
- NumPy/SciPy: Standard scientific computing
- SymPy: Symbolic mathematics
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

from .base import BackendBase
from .errors import BackendError, NotImplementedInBackend
from .context import use_backend
from .vectorize import vmap
from .optimize import minimize, root, curve_fit


_CURRENT_BACKEND = None
_JAX_BACKEND = None
_NUMPY_BACKEND = None
_SYMPY_BACKEND = None


def get_backend(name: Optional[Literal["jax", "numpy", "sympy"]] = None) -> BackendBase:
    """Get the backend with the given name.

    Args:
        name: The name of the backend to get. If None, returns the current backend.

    Returns:
        The requested backend.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    global _CURRENT_BACKEND, _JAX_BACKEND, _NUMPY_BACKEND, _SYMPY_BACKEND

    if name is None:
        if _CURRENT_BACKEND is None:
            # Default to JAX if available, otherwise NumPy
            try:
                return get_backend("jax")
            except (ImportError, BackendError):
                return get_backend("numpy")
        return _CURRENT_BACKEND

    if name.lower() == "jax":
        if _JAX_BACKEND is None:
            from .jax_backend import JAXBackend
            _JAX_BACKEND = JAXBackend()
        _CURRENT_BACKEND = _JAX_BACKEND
        return _JAX_BACKEND

    if name.lower() == "numpy":
        if _NUMPY_BACKEND is None:
            from .numpy_backend import NumPyBackend
            _NUMPY_BACKEND = NumPyBackend()
        _CURRENT_BACKEND = _NUMPY_BACKEND
        return _NUMPY_BACKEND

    if name.lower() == "sympy":
        if _SYMPY_BACKEND is None:
            from .sympy_backend import SymPyBackend
            _SYMPY_BACKEND = SymPyBackend()
        _CURRENT_BACKEND = _SYMPY_BACKEND
        return _SYMPY_BACKEND

    raise ValueError(f"Unsupported backend: {name}. Available backends: jax, numpy, sympy")


def set_backend(name: Literal["jax", "numpy", "sympy"]) -> None:
    """Set the default backend.

    Args:
        name: The name of the backend to set as default.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    get_backend(name)  # This will update _CURRENT_BACKEND


# Initialize backend from environment variable if set
if "ZFIT_BACKEND" in os.environ:
    try:
        set_backend(os.environ["ZFIT_BACKEND"].lower())
    except (ImportError, ValueError, BackendError) as e:
        import warnings
        warnings.warn(f"Failed to set backend from ZFIT_BACKEND environment variable: {e}")

# Create a default backend
numpy = get_backend()

# Expose common functions directly
array = numpy.array
asarray = numpy.asarray
zeros = numpy.zeros
ones = numpy.ones
full = numpy.full
sum = numpy.sum
exp = numpy.exp
log = numpy.log
sin = numpy.sin
cos = numpy.cos
tan = numpy.tan
arcsin = numpy.arcsin
arccos = numpy.arccos
arctan = numpy.arctan
sinh = numpy.sinh
cosh = numpy.cosh
tanh = numpy.tanh
arcsinh = numpy.arcsinh
arccosh = numpy.arccosh
arctanh = numpy.arctanh
power = numpy.power
sqrt = numpy.sqrt
square = numpy.square
absolute = numpy.absolute
abs = numpy.absolute
mean = numpy.mean
var = numpy.var
std = numpy.std
min = numpy.min
max = numpy.max
argmin = numpy.argmin
argmax = numpy.argmax
clip = numpy.clip
round = numpy.round
dot = numpy.dot
tensordot = numpy.tensordot
matmul = numpy.matmul

# Random functions
normal = numpy.normal
uniform = numpy.uniform
random_split = numpy.random_split

# Linear algebra
inv = numpy.inv
eigh = numpy.eigh
cholesky = numpy.cholesky
solve = numpy.solve

# Differential operations
grad = numpy.grad
value_and_grad = numpy.value_and_grad
hessian = numpy.hessian
jacobian = numpy.jacobian
custom_jvp = numpy.custom_jvp
custom_vjp = numpy.custom_vjp

# Array manipulation
reshape = numpy.reshape
transpose = numpy.transpose
concatenate = numpy.concatenate
stack = numpy.stack
vstack = numpy.vstack
hstack = numpy.hstack
where = numpy.where

# Control flow operations
scan = numpy.scan

# PyTree operations
tree_map = numpy.tree_map
tree_flatten = numpy.tree_flatten
tree_unflatten = numpy.tree_unflatten

# Transformations
jit = numpy.jit
vmap = numpy.vmap
pmap = numpy.pmap
checkpoint = numpy.checkpoint

# Other functions
sign = numpy.sign
floor = numpy.floor
ceil = numpy.ceil

__all__ = [
    # Backend management
    "get_backend", "set_backend", "use_backend", "numpy", "BackendError", "NotImplementedInBackend",

    # Array creation
    "array", "asarray", "zeros", "ones", "full",

    # Math operations
    "sum", "exp", "log", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "power", "sqrt",
    "square", "absolute", "abs", "mean", "var", "std", "min", "max", "argmin",
    "argmax", "clip", "round", "dot", "tensordot", "matmul",

    # Random functions
    "normal", "uniform", "random_split",

    # Linear algebra
    "inv", "eigh", "cholesky", "solve",

    # Differential operations
    "grad", "value_and_grad", "hessian", "jacobian", "custom_jvp", "custom_vjp",

    # Array manipulation
    "reshape", "transpose", "concatenate", "stack", "vstack", "hstack", "where",

    # Control flow operations
    "scan",

    # PyTree operations
    "tree_map", "tree_flatten", "tree_unflatten",

    # Transformations
    "jit", "vmap", "pmap", "checkpoint",

    # Optimization
    "minimize", "root", "curve_fit",

    # Other functions
    "sign", "floor", "ceil"
]