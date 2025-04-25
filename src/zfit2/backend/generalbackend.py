# """Backend for zfit2.
#
# This module provides a unified interface to different computational backends.
# """
# from __future__ import annotations
#
# import importlib.util
# from typing import Any, Callable, Optional, Sequence, Tuple, Union
#
# # from .base import BackendBase
# # from .errors import BackendError, NotImplementedInBackend
#
#
# class Backend(BackendBase):
#     """Backend for zfit2 - compatibility layer for existing code.
#
#     This class is maintained for backwards compatibility and will be
#     replaced by the new backend system in the future.
#     """
#
#     def __init__(self, backend_name: str = None):
#         """Initialize with the specified backend.
#
#         Args:
#             backend_name: The name of the backend to use.
#                 If None, JAX will be used if available, otherwise NumPy.
#         """
#         from . import get_backend
#         self._backend = get_backend(backend_name)
#
#     @property
#     def name(self) -> str:
#         """Return the name of the backend."""
#         return self._backend.name
#
#     def __repr__(self) -> str:
#         return f"{self._backend.name}Backend"
#
#     def __str__(self) -> str:
#         return f"{self._backend.name}Backend"
#
#     def __getattr__(self, name: str) -> Any:
#         """Get an attribute from the backend."""
#         return getattr(self._backend, name)
#
#     # Core array creation functions
#     def array(self, obj, dtype=None, copy=None) -> Any:
#         """Create an array."""
#         return self._backend.array(obj, dtype=dtype, copy=copy)
#
#     def asarray(self, a, dtype=None, copy=None) -> Any:
#         """Convert the input to an array."""
#         return self._backend.asarray(a, dtype=dtype, copy=copy)
#
#     def zeros(self, shape, dtype=None) -> Any:
#         """Return a new array of given shape and type, filled with zeros."""
#         return self._backend.zeros(shape, dtype=dtype)
#
#     def ones(self, shape, dtype=None) -> Any:
#         """Return a new array of given shape and type, filled with ones."""
#         return self._backend.ones(shape, dtype=dtype)
#
#     def full(self, shape, fill_value, dtype=None) -> Any:
#         """Return a new array of given shape and type, filled with fill_value."""
#         return self._backend.full(shape, fill_value, dtype=dtype)
#
#     # Math operations
#     def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
#         """Sum of array elements over given axes."""
#         return self._backend.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)
#
#     def exp(self, x) -> Any:
#         """Calculate the exponential of all elements in the input array."""
#         return self._backend.exp(x)
#
#     def log(self, x) -> Any:
#         """Natural logarithm, element-wise."""
#         return self._backend.log(x)
#
#     def sin(self, x) -> Any:
#         """Sine, element-wise."""
#         return self._backend.sin(x)
#
#     def cos(self, x) -> Any:
#         """Cosine, element-wise."""
#         return self._backend.cos(x)
#
#     def tan(self, x) -> Any:
#         """Tangent, element-wise."""
#         return self._backend.tan(x)
#
#     def arcsin(self, x) -> Any:
#         """Inverse sine, element-wise."""
#         return self._backend.arcsin(x)
#
#     def arccos(self, x) -> Any:
#         """Inverse cosine, element-wise."""
#         return self._backend.arccos(x)
#
#     def arctan(self, x) -> Any:
#         """Inverse tangent, element-wise."""
#         return self._backend.arctan(x)
#
#     def sinh(self, x) -> Any:
#         """Hyperbolic sine, element-wise."""
#         return self._backend.sinh(x)
#
#     def cosh(self, x) -> Any:
#         """Hyperbolic cosine, element-wise."""
#         return self._backend.cosh(x)
#
#     def tanh(self, x) -> Any:
#         """Hyperbolic tangent, element-wise."""
#         return self._backend.tanh(x)
#
#     def arcsinh(self, x) -> Any:
#         """Inverse hyperbolic sine, element-wise."""
#         return self._backend.arcsinh(x)
#
#     def arccosh(self, x) -> Any:
#         """Inverse hyperbolic cosine, element-wise."""
#         return self._backend.arccosh(x)
#
#     def arctanh(self, x) -> Any:
#         """Inverse hyperbolic tangent, element-wise."""
#         return self._backend.arctanh(x)
#
#     def power(self, x1, x2) -> Any:
#         """First array elements raised to powers from second array, element-wise."""
#         return self._backend.power(x1, x2)
#
#     def sqrt(self, x) -> Any:
#         """Return the non-negative square-root of an array, element-wise."""
#         return self._backend.sqrt(x)
#
#     def square(self, x) -> Any:
#         """Return the element-wise square of the input."""
#         return self._backend.square(x)
#
#     def absolute(self, x) -> Any:
#         """Calculate the absolute value element-wise."""
#         return self._backend.absolute(x)
#
#     def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
#         """Compute the arithmetic mean along the specified axis."""
#         return self._backend.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
#
#     def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
#         """Compute the variance along the specified axis."""
#         return self._backend.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
#
#     def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
#         """Compute the standard deviation along the specified axis."""
#         return self._backend.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
#
#     def min(self, a, axis=None, keepdims=False) -> Any:
#         """Return minimum of an array or minimum along an axis."""
#         return self._backend.min(a, axis=axis, keepdims=keepdims)
#
#     def max(self, a, axis=None, keepdims=False) -> Any:
#         """Return maximum of an array or maximum along an axis."""
#         return self._backend.max(a, axis=axis, keepdims=keepdims)
#
#     def argmin(self, a, axis=None) -> Any:
#         """Return indices of the minimum values along the given axis."""
#         return self._backend.argmin(a, axis=axis)
#
#     def argmax(self, a, axis=None) -> Any:
#         """Return indices of the maximum values along the given axis."""
#         return self._backend.argmax(a, axis=axis)
#
#     def clip(self, a, a_min, a_max) -> Any:
#         """Clip (limit) the values in an array."""
#         return self._backend.clip(a, a_min, a_max)
#
#     def round(self, a, decimals=0) -> Any:
#         """Round an array to the given number of decimals."""
#         return self._backend.round(a, decimals=decimals)
#
#     def dot(self, a, b) -> Any:
#         """Dot product of two arrays."""
#         return self._backend.dot(a, b)
#
#     def tensordot(self, a, b, axes=2) -> Any:
#         """Compute tensor dot product along specified axes."""
#         return self._backend.tensordot(a, b, axes=axes)
#
#     def matmul(self, a, b) -> Any:
#         """Matrix product of two arrays."""
#         return self._backend.matmul(a, b)
#
#     # Statistical functions
#     def normal(self, loc=0.0, scale=1.0, size=None, dtype=None, key=None) -> Any:
#         """Draw random samples from a normal distribution."""
#         return self._backend.normal(loc=loc, scale=scale, size=size, dtype=dtype, key=key)
#
#     def uniform(self, low=0.0, high=1.0, size=None, dtype=None, key=None) -> Any:
#         """Draw random samples from a uniform distribution."""
#         return self._backend.uniform(low=low, high=high, size=size, dtype=dtype, key=key)
#
#     # Linear algebra operations
#     def inv(self, a) -> Any:
#         """Compute the inverse of a matrix."""
#         return self._backend.inv(a)
#
#     def eigh(self, a) -> Tuple[Any, Any]:
#         """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
#         return self._backend.eigh(a)
#
#     def cholesky(self, a) -> Any:
#         """Cholesky decomposition."""
#         return self._backend.cholesky(a)
#
#     def solve(self, a, b) -> Any:
#         """Solve a linear matrix equation, or system of linear scalar equations."""
#         return self._backend.solve(a, b)
#
#     # Differential operations
#     def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
#         """Return a function to compute the gradient of `fun` with respect to positional arguments."""
#         return self._backend.grad(fun, argnums=argnums)
#
#     def hessian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
#         """Return a function to compute the Hessian of `fun` with respect to positional arguments."""
#         return self._backend.hessian(fun, argnums=argnums)
#
#     def jacobian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
#         """Return a function to compute the Jacobian of `fun` with respect to positional arguments."""
#         return self._backend.jacobian(fun, argnums=argnums)
#
#     # Array manipulation
#     def reshape(self, a, newshape) -> Any:
#         """Gives a new shape to an array without changing its data."""
#         return self._backend.reshape(a, newshape)
#
#     def transpose(self, a, axes=None) -> Any:
#         """Permute the dimensions of an array."""
#         return self._backend.transpose(a, axes=axes)
#
#     def concatenate(self, arrays, axis=0) -> Any:
#         """Join a sequence of arrays along an existing axis."""
#         return self._backend.concatenate(arrays, axis=axis)
#
#     def stack(self, arrays, axis=0) -> Any:
#         """Join a sequence of arrays along a new axis."""
#         return self._backend.stack(arrays, axis=axis)
#
#     def vstack(self, tup) -> Any:
#         """Stack arrays in sequence vertically (row wise)."""
#         return self._backend.vstack(tup)
#
#     def hstack(self, tup) -> Any:
#         """Stack arrays in sequence horizontally (column wise)."""
#         return self._backend.hstack(tup)
#
#     def where(self, condition, x=None, y=None) -> Any:
#         """Return elements chosen from `x` or `y` depending on `condition`."""
#         return self._backend.where(condition, x, y)
#
#     # Other standard functions
#     def sign(self, x) -> Any:
#         """Returns an element-wise indication of the sign of a number."""
#         return self._backend.sign(x)
#
#     def floor(self, x) -> Any:
#         """Return the floor of the input, element-wise."""
#         return self._backend.floor(x)
#
#     def ceil(self, x) -> Any:
#         """Return the ceiling of the input, element-wise."""
#         return self._backend.ceil(x)