"""JAX backend implementation."""
from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from .base import BackendBase
from .errors import NotImplementedInBackend

# Type variables for PyTree operations
T = TypeVar('T')
U = TypeVar('U')

class JAXBackend(BackendBase):
    """Backend implementation using JAX."""
    
    def __init__(self):
        """Initialize the JAX backend."""
        if not importlib.util.find_spec("jax"):
            raise ImportError(
                "JAX is not installed. Please install it with `pip install jax`."
            )
        
        import jax
        import jax.numpy as jnp
        import jax.lax as lax
        from jax import (
            grad, hessian, jacfwd, jacrev, jit, vmap, pmap, 
            value_and_grad, custom_jvp, custom_vjp, checkpoint,
            tree_map, tree_flatten, tree_unflatten
        )
        
        self._jax = jax
        self._jnp = jnp
        self._lax = lax
        self._grad = grad
        self._hessian = hessian
        self._jacfwd = jacfwd
        self._jacrev = jacrev
        self._jit = jit
        self._vmap = vmap
        self._pmap = pmap
        self._value_and_grad = value_and_grad
        self._custom_jvp = custom_jvp
        self._custom_vjp = custom_vjp
        self._checkpoint = checkpoint
        self._tree_map = tree_map
        self._tree_flatten = tree_flatten
        self._tree_unflatten = tree_unflatten
    
    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return "JAX"
    
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the JAX numpy module."""
        try:
            return getattr(self._jnp, name)
        except AttributeError:
            try:
                return getattr(self._jax, name)
            except AttributeError:
                try:
                    return getattr(self._lax, name)
                except AttributeError:
                    self._not_implemented(name)
    
    # Core array creation functions
    def array(self, obj, dtype=None, copy=None, device=None) -> Any:
        """Create a JAX array."""
        if device is not None:
            with self._jax.default_device(device):
                return self._jnp.array(obj, dtype=dtype)
        return self._jnp.array(obj, dtype=dtype)
    
    def asarray(self, a, dtype=None, copy=None, device=None) -> Any:
        """Convert the input to a JAX array."""
        if device is not None:
            with self._jax.default_device(device):
                return self._jnp.asarray(a, dtype=dtype)
        return self._jnp.asarray(a, dtype=dtype)
    
    def zeros(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with zeros."""
        if device is not None:
            with self._jax.default_device(device):
                return self._jnp.zeros(shape, dtype=dtype)
        return self._jnp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with ones."""
        if device is not None:
            with self._jax.default_device(device):
                return self._jnp.ones(shape, dtype=dtype)
        return self._jnp.ones(shape, dtype=dtype)
    
    def full(self, shape, fill_value, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with fill_value."""
        if device is not None:
            with self._jax.default_device(device):
                return self._jnp.full(shape, fill_value, dtype=dtype)
        return self._jnp.full(shape, fill_value, dtype=dtype)
    
    # Math operations
    def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Sum of array elements over given axes."""
        return self._jnp.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)
    
    def exp(self, x) -> Any:
        """Calculate the exponential of all elements in the input array."""
        return self._jnp.exp(x)
    
    def log(self, x) -> Any:
        """Natural logarithm, element-wise."""
        return self._jnp.log(x)
    
    def sin(self, x) -> Any:
        """Sine, element-wise."""
        return self._jnp.sin(x)
    
    def cos(self, x) -> Any:
        """Cosine, element-wise."""
        return self._jnp.cos(x)
    
    def tan(self, x) -> Any:
        """Tangent, element-wise."""
        return self._jnp.tan(x)
    
    def arcsin(self, x) -> Any:
        """Inverse sine, element-wise."""
        return self._jnp.arcsin(x)
    
    def arccos(self, x) -> Any:
        """Inverse cosine, element-wise."""
        return self._jnp.arccos(x)
    
    def arctan(self, x) -> Any:
        """Inverse tangent, element-wise."""
        return self._jnp.arctan(x)
    
    def sinh(self, x) -> Any:
        """Hyperbolic sine, element-wise."""
        return self._jnp.sinh(x)
    
    def cosh(self, x) -> Any:
        """Hyperbolic cosine, element-wise."""
        return self._jnp.cosh(x)
    
    def tanh(self, x) -> Any:
        """Hyperbolic tangent, element-wise."""
        return self._jnp.tanh(x)
    
    def arcsinh(self, x) -> Any:
        """Inverse hyperbolic sine, element-wise."""
        return self._jnp.arcsinh(x)
    
    def arccosh(self, x) -> Any:
        """Inverse hyperbolic cosine, element-wise."""
        return self._jnp.arccosh(x)
    
    def arctanh(self, x) -> Any:
        """Inverse hyperbolic tangent, element-wise."""
        return self._jnp.arctanh(x)
    
    def power(self, x1, x2) -> Any:
        """First array elements raised to powers from second array, element-wise."""
        return self._jnp.power(x1, x2)
    
    def sqrt(self, x) -> Any:
        """Return the non-negative square-root of an array, element-wise."""
        return self._jnp.sqrt(x)
    
    def square(self, x) -> Any:
        """Return the element-wise square of the input."""
        return self._jnp.square(x)
    
    def absolute(self, x) -> Any:
        """Calculate the absolute value element-wise."""
        return self._jnp.absolute(x)
    
    def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Compute the arithmetic mean along the specified axis."""
        return self._jnp.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
    
    def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the variance along the specified axis."""
        return self._jnp.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    
    def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the standard deviation along the specified axis."""
        return self._jnp.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    
    def min(self, a, axis=None, keepdims=False) -> Any:
        """Return minimum of an array or minimum along an axis."""
        return self._jnp.min(a, axis=axis, keepdims=keepdims)
    
    def max(self, a, axis=None, keepdims=False) -> Any:
        """Return maximum of an array or maximum along an axis."""
        return self._jnp.max(a, axis=axis, keepdims=keepdims)
    
    def argmin(self, a, axis=None) -> Any:
        """Return indices of the minimum values along the given axis."""
        return self._jnp.argmin(a, axis=axis)
    
    def argmax(self, a, axis=None) -> Any:
        """Return indices of the maximum values along the given axis."""
        return self._jnp.argmax(a, axis=axis)
    
    def clip(self, a, a_min, a_max) -> Any:
        """Clip (limit) the values in an array."""
        return self._jnp.clip(a, a_min, a_max)
    
    def round(self, a, decimals=0) -> Any:
        """Round an array to the given number of decimals."""
        return self._jnp.round(a, decimals=decimals)
    
    def dot(self, a, b) -> Any:
        """Dot product of two arrays."""
        return self._jnp.dot(a, b)
    
    def tensordot(self, a, b, axes=2) -> Any:
        """Compute tensor dot product along specified axes."""
        return self._jnp.tensordot(a, b, axes=axes)
    
    def matmul(self, a, b) -> Any:
        """Matrix product of two arrays."""
        return self._jnp.matmul(a, b)
    
    # Statistical functions
    def normal(self, key=None, shape=None, dtype=None, loc=0.0, scale=1.0) -> Any:
        """Draw random samples from a normal distribution.
        
        This follows JAX's random API where the key is the first parameter.
        For compatibility with numpy, we also accept named parameters in a different order.
        """
        if key is None:
            key = self._jax.random.key(0)
        if shape is None:
            shape = ()
        return self._jax.random.normal(key, shape=shape, dtype=dtype) * scale + loc
    
    def uniform(self, key=None, shape=None, dtype=None, minval=0.0, maxval=1.0) -> Any:
        """Draw random samples from a uniform distribution.
        
        This follows JAX's random API where the key is the first parameter.
        For compatibility with numpy, we also accept named parameters in a different order.
        """
        if key is None:
            key = self._jax.random.key(0)
        if shape is None:
            shape = ()
        return self._jax.random.uniform(key, shape=shape, dtype=dtype, minval=minval, maxval=maxval)
    
    def random_split(self, key, num=2) -> Any:
        """Split a PRNG key into multiple keys."""
        return self._jax.random.split(key, num=num)
    
    # Linear algebra operations
    def inv(self, a) -> Any:
        """Compute the inverse of a matrix."""
        return self._jnp.linalg.inv(a)
    
    def eigh(self, a) -> Tuple[Any, Any]:
        """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        return self._jnp.linalg.eigh(a)
    
    def cholesky(self, a) -> Any:
        """Cholesky decomposition."""
        return self._jnp.linalg.cholesky(a)
    
    def solve(self, a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""
        return self._jnp.linalg.solve(a, b)
    
    # Differential operations
    def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the gradient of `fun` with respect to positional arguments."""
        return self._grad(fun, argnums=argnums)
    
    def value_and_grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function that evaluates both fun and its gradient."""
        return self._value_and_grad(fun, argnums=argnums)
    
    def hessian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the Hessian of `fun` with respect to positional arguments."""
        return self._hessian(fun, argnums=argnums)
    
    def jacobian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the Jacobian of `fun` with respect to positional arguments."""
        return self._jacrev(fun, argnums=argnums)
    
    def custom_jvp(self, fun: Callable, jvp: Optional[Callable] = None) -> Callable:
        """Specify a custom JVP rule for a function."""
        if jvp is None:
            return self._custom_jvp(fun)
        
        fun_jvp = self._custom_jvp(fun)
        fun_jvp.defjvp(jvp)
        return fun_jvp
    
    def custom_vjp(self, fun: Callable, fwd: Optional[Callable] = None, bwd: Optional[Callable] = None) -> Callable:
        """Specify a custom VJP rule for a function."""
        if fwd is None or bwd is None:
            return self._custom_vjp(fun)
        
        fun_vjp = self._custom_vjp(fun)
        fun_vjp.defvjp(fwd, bwd)
        return fun_vjp
    
    # Array manipulation
    def reshape(self, a, newshape) -> Any:
        """Gives a new shape to an array without changing its data."""
        return self._jnp.reshape(a, newshape)
    
    def transpose(self, a, axes=None) -> Any:
        """Permute the dimensions of an array."""
        return self._jnp.transpose(a, axes=axes)
    
    def concatenate(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along an existing axis."""
        return self._jnp.concatenate(arrays, axis=axis)
    
    def stack(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along a new axis."""
        return self._jnp.stack(arrays, axis=axis)
    
    def vstack(self, tup) -> Any:
        """Stack arrays in sequence vertically (row wise)."""
        return self._jnp.vstack(tup)
    
    def hstack(self, tup) -> Any:
        """Stack arrays in sequence horizontally (column wise)."""
        return self._jnp.hstack(tup)
    
    def where(self, condition, x=None, y=None) -> Any:
        """Return elements chosen from `x` or `y` depending on `condition`."""
        return self._jnp.where(condition, x, y)
    
    # Control flow operations
    def scan(self, f, init, xs, length=None, reverse=False) -> Any:
        """Scan a function over leading array axes while carrying along state."""
        return self._lax.scan(f, init, xs, length=length, reverse=reverse)
    
    # PyTree operations
    def tree_map(self, f: Callable[[T], U], tree: T, *rest: Any) -> U:
        """Map a function over a pytree."""
        return self._tree_map(f, tree, *rest)
    
    def tree_flatten(self, tree: T) -> Tuple[List[Any], Any]:
        """Flatten a pytree into a list of leaves and a treedef."""
        return self._tree_flatten(tree)
    
    def tree_unflatten(self, treedef: Any, leaves: List[Any]) -> Any:
        """Unflatten a list of leaves and a treedef into a pytree."""
        return self._tree_unflatten(treedef, leaves)
    
    # Transformations
    def jit(self, fun: Callable, static_argnums: Union[int, Sequence[int]] = ()) -> Callable:
        """JIT-compile a function for faster execution."""
        return self._jit(fun, static_argnums=static_argnums)
    
    def vmap(self, fun: Callable, in_axes=0, out_axes=0) -> Callable:
        """Vectorize a function along the specified axes."""
        return self._vmap(fun, in_axes=in_axes, out_axes=out_axes)
    
    def pmap(self, fun: Callable, axis_name=None, devices=None) -> Callable:
        """Parallel map over an axis."""
        return self._pmap(fun, axis_name=axis_name, devices=devices)
    
    def checkpoint(self, fun: Callable, concrete: bool = False) -> Callable:
        """Checkpoint a function to save memory during backpropagation."""
        return self._checkpoint(fun, concrete=concrete)
    
    # Other standard functions
    def sign(self, x) -> Any:
        """Returns an element-wise indication of the sign of a number."""
        return self._jnp.sign(x)
        
    def floor(self, x) -> Any:
        """Return the floor of the input, element-wise."""
        return self._jnp.floor(x)
    
    def ceil(self, x) -> Any:
        """Return the ceiling of the input, element-wise."""
        return self._jnp.ceil(x)
