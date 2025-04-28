"""Base class for backend implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from .errors import NotImplementedInBackend

# Type variables for PyTree operations
T = TypeVar('T')
U = TypeVar('U')

class BackendBase(ABC):
    """Base class for all backend implementations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""
        pass
    
    @abstractmethod
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the backend module."""
        pass
    
    # Core array creation functions
    @abstractmethod
    def array(self, obj, dtype=None, copy=None, device=None) -> Any:
        """Create an array."""
        pass
    
    @abstractmethod
    def asarray(self, a, dtype=None, copy=None, device=None) -> Any:
        """Convert the input to an array."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with zeros."""
        pass
    
    @abstractmethod
    def ones(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with ones."""
        pass
    
    @abstractmethod
    def full(self, shape, fill_value, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with fill_value."""
        pass
    
    # Math operations
    @abstractmethod
    def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Sum of array elements over given axes."""
        pass
    
    @abstractmethod
    def exp(self, x) -> Any:
        """Calculate the exponential of all elements in the input array."""
        pass
    
    @abstractmethod
    def log(self, x) -> Any:
        """Natural logarithm, element-wise."""
        pass
    
    @abstractmethod
    def sin(self, x) -> Any:
        """Sine, element-wise."""
        pass
    
    @abstractmethod
    def cos(self, x) -> Any:
        """Cosine, element-wise."""
        pass
    
    @abstractmethod
    def tan(self, x) -> Any:
        """Tangent, element-wise."""
        pass
    
    @abstractmethod
    def arcsin(self, x) -> Any:
        """Inverse sine, element-wise."""
        pass
    
    @abstractmethod
    def arccos(self, x) -> Any:
        """Inverse cosine, element-wise."""
        pass
    
    @abstractmethod
    def arctan(self, x) -> Any:
        """Inverse tangent, element-wise."""
        pass
    
    @abstractmethod
    def sinh(self, x) -> Any:
        """Hyperbolic sine, element-wise."""
        pass
    
    @abstractmethod
    def cosh(self, x) -> Any:
        """Hyperbolic cosine, element-wise."""
        pass
    
    @abstractmethod
    def tanh(self, x) -> Any:
        """Hyperbolic tangent, element-wise."""
        pass
    
    @abstractmethod
    def arcsinh(self, x) -> Any:
        """Inverse hyperbolic sine, element-wise."""
        pass
    
    @abstractmethod
    def arccosh(self, x) -> Any:
        """Inverse hyperbolic cosine, element-wise."""
        pass
    
    @abstractmethod
    def arctanh(self, x) -> Any:
        """Inverse hyperbolic tangent, element-wise."""
        pass
    
    @abstractmethod
    def power(self, x1, x2) -> Any:
        """First array elements raised to powers from second array, element-wise."""
        pass
    
    @abstractmethod
    def sqrt(self, x) -> Any:
        """Return the non-negative square-root of an array, element-wise."""
        pass
    
    @abstractmethod
    def square(self, x) -> Any:
        """Return the element-wise square of the input."""
        pass
    
    @abstractmethod
    def absolute(self, x) -> Any:
        """Calculate the absolute value element-wise."""
        pass
    
    @abstractmethod
    def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Compute the arithmetic mean along the specified axis."""
        pass
    
    @abstractmethod
    def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the variance along the specified axis."""
        pass
    
    @abstractmethod
    def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the standard deviation along the specified axis."""
        pass
    
    @abstractmethod
    def min(self, a, axis=None, keepdims=False) -> Any:
        """Return minimum of an array or minimum along an axis."""
        pass
    
    @abstractmethod
    def max(self, a, axis=None, keepdims=False) -> Any:
        """Return maximum of an array or maximum along an axis."""
        pass
    
    @abstractmethod
    def argmin(self, a, axis=None) -> Any:
        """Return indices of the minimum values along the given axis."""
        pass
    
    @abstractmethod
    def argmax(self, a, axis=None) -> Any:
        """Return indices of the maximum values along the given axis."""
        pass
    
    @abstractmethod
    def clip(self, a, a_min, a_max) -> Any:
        """Clip (limit) the values in an array."""
        pass
    
    @abstractmethod
    def round(self, a, decimals=0) -> Any:
        """Round an array to the given number of decimals."""
        pass
    
    @abstractmethod
    def dot(self, a, b) -> Any:
        """Dot product of two arrays."""
        pass
    
    @abstractmethod
    def tensordot(self, a, b, axes=2) -> Any:
        """Compute tensor dot product along specified axes."""
        pass
    
    @abstractmethod
    def matmul(self, a, b) -> Any:
        """Matrix product of two arrays."""
        pass
    
    # Statistical functions
    @abstractmethod
    def normal(self, key=None, shape=None, dtype=None, loc=0.0, scale=1.0) -> Any:
        """Draw random samples from a normal distribution."""
        pass
    
    @abstractmethod
    def uniform(self, key=None, shape=None, dtype=None, minval=0.0, maxval=1.0) -> Any:
        """Draw random samples from a uniform distribution."""
        pass
    
    @abstractmethod
    def random_split(self, key, num=2) -> Any:
        """Split a PRNG key into multiple keys."""
        pass
    
    # Linear algebra operations
    @abstractmethod
    def inv(self, a) -> Any:
        """Compute the inverse of a matrix."""
        pass
    
    @abstractmethod
    def eigh(self, a) -> Tuple[Any, Any]:
        """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        pass
    
    @abstractmethod
    def cholesky(self, a) -> Any:
        """Cholesky decomposition."""
        pass
    
    @abstractmethod
    def solve(self, a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""
        pass
    
    # Differential operations
    @abstractmethod
    def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the gradient of `fun` with respect to positional arguments."""
        pass
    
    @abstractmethod
    def value_and_grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function that evaluates both fun and its gradient."""
        pass
    
    @abstractmethod
    def hessian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the Hessian of `fun` with respect to positional arguments."""
        pass
    
    @abstractmethod
    def jacobian(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the Jacobian of `fun` with respect to positional arguments."""
        pass
    
    @abstractmethod
    def custom_jvp(self, fun: Callable, jvp: Optional[Callable] = None) -> Callable:
        """Specify a custom JVP rule for a function."""
        pass
    
    @abstractmethod
    def custom_vjp(self, fun: Callable, fwd: Optional[Callable] = None, bwd: Optional[Callable] = None) -> Callable:
        """Specify a custom VJP rule for a function."""
        pass
    
    # Array manipulation
    @abstractmethod
    def reshape(self, a, newshape) -> Any:
        """Gives a new shape to an array without changing its data."""
        pass
    
    @abstractmethod
    def transpose(self, a, axes=None) -> Any:
        """Permute the dimensions of an array."""
        pass
    
    @abstractmethod
    def concatenate(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along an existing axis."""
        pass
    
    @abstractmethod
    def stack(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along a new axis."""
        pass
    
    @abstractmethod
    def vstack(self, tup) -> Any:
        """Stack arrays in sequence vertically (row wise)."""
        pass
    
    @abstractmethod
    def hstack(self, tup) -> Any:
        """Stack arrays in sequence horizontally (column wise)."""
        pass
    
    @abstractmethod
    def where(self, condition, x=None, y=None) -> Any:
        """Return elements chosen from `x` or `y` depending on `condition`."""
        pass
    
    # Control flow operations
    @abstractmethod
    def scan(self, f, init, xs, length=None, reverse=False) -> Any:
        """Scan a function over leading array axes while carrying along state."""
        pass
    
    # PyTree operations
    @abstractmethod
    def tree_map(self, f: Callable[[T], U], tree: T, *rest: Any) -> U:
        """Map a function over a pytree."""
        pass
    
    @abstractmethod
    def tree_flatten(self, tree: T) -> Tuple[List[Any], Any]:
        """Flatten a pytree into a list of leaves and a treedef."""
        pass
    
    @abstractmethod
    def tree_unflatten(self, treedef: Any, leaves: List[Any]) -> Any:
        """Unflatten a list of leaves and a treedef into a pytree."""
        pass
    
    # Transformations
    @abstractmethod
    def jit(self, fun: Callable, static_argnums: Union[int, Sequence[int]] = ()) -> Callable:
        """JIT-compile a function for faster execution."""
        pass
    
    @abstractmethod
    def vmap(self, fun: Callable, in_axes=0, out_axes=0) -> Callable:
        """Vectorize a function along the specified axes."""
        pass
    
    @abstractmethod
    def pmap(self, fun: Callable, axis_name=None, devices=None) -> Callable:
        """Parallel map over an axis."""
        pass
    
    @abstractmethod
    def checkpoint(self, fun: Callable, concrete: bool = False) -> Callable:
        """Checkpoint a function to save memory during backpropagation."""
        pass
    
    # Other standard functions
    @abstractmethod
    def sign(self, x) -> Any:
        """Returns an element-wise indication of the sign of a number."""
        pass
        
    @abstractmethod
    def floor(self, x) -> Any:
        """Return the floor of the input, element-wise."""
        pass
    
    @abstractmethod
    def ceil(self, x) -> Any:
        """Return the ceiling of the input, element-wise."""
        pass
    
    # Helper methods
    def _not_implemented(self, func_name: str) -> Any:
        """Raise NotImplementedInBackend error."""
        raise NotImplementedInBackend(func_name, self.name)
