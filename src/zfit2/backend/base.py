"""Base class for backend implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

from .errors import NotImplementedInBackend

# Type variables for PyTree operations
T = TypeVar("T")
U = TypeVar("U")


class BackendBase(ABC):
    """Base class for all backend implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend."""

    @abstractmethod
    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the backend module."""

    # Core array creation functions
    @abstractmethod
    def array(self, obj, dtype=None, copy=None, device=None) -> Any:
        """Create an array."""

    @abstractmethod
    def asarray(self, a, dtype=None, copy=None, device=None) -> Any:
        """Convert the input to an array."""

    @abstractmethod
    def zeros(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with zeros."""

    @abstractmethod
    def ones(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with ones."""

    @abstractmethod
    def full(self, shape, fill_value, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with fill_value."""

    # Array information functions
    def shape(self, a) -> Any:
        """Return the shape of an array."""
        return a.shape if hasattr(a, "shape") else ()

    def size(self, a) -> Any:
        """Return the number of elements in an array."""
        return a.size if hasattr(a, "size") else 1

    def ndim(self, a) -> Any:
        """Return the number of dimensions of an array."""
        return a.ndim if hasattr(a, "ndim") else 0

    def dtype(self, a) -> Any:
        """Return the data type of an array."""
        return a.dtype if hasattr(a, "dtype") else type(a)

    # Math operations
    @abstractmethod
    def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Sum of array elements over given axes."""

    @abstractmethod
    def exp(self, x) -> Any:
        """Calculate the exponential of all elements in the input array."""

    @abstractmethod
    def log(self, x) -> Any:
        """Natural logarithm, element-wise."""

    @abstractmethod
    def sin(self, x) -> Any:
        """Sine, element-wise."""

    @abstractmethod
    def cos(self, x) -> Any:
        """Cosine, element-wise."""

    @abstractmethod
    def tan(self, x) -> Any:
        """Tangent, element-wise."""

    @abstractmethod
    def arcsin(self, x) -> Any:
        """Inverse sine, element-wise."""

    @abstractmethod
    def arccos(self, x) -> Any:
        """Inverse cosine, element-wise."""

    @abstractmethod
    def arctan(self, x) -> Any:
        """Inverse tangent, element-wise."""

    @abstractmethod
    def sinh(self, x) -> Any:
        """Hyperbolic sine, element-wise."""

    @abstractmethod
    def cosh(self, x) -> Any:
        """Hyperbolic cosine, element-wise."""

    @abstractmethod
    def tanh(self, x) -> Any:
        """Hyperbolic tangent, element-wise."""

    @abstractmethod
    def arcsinh(self, x) -> Any:
        """Inverse hyperbolic sine, element-wise."""

    @abstractmethod
    def arccosh(self, x) -> Any:
        """Inverse hyperbolic cosine, element-wise."""

    @abstractmethod
    def arctanh(self, x) -> Any:
        """Inverse hyperbolic tangent, element-wise."""

    @abstractmethod
    def power(self, x1, x2) -> Any:
        """First array elements raised to powers from second array, element-wise."""

    @abstractmethod
    def sqrt(self, x) -> Any:
        """Return the non-negative square-root of an array, element-wise."""

    @abstractmethod
    def square(self, x) -> Any:
        """Return the element-wise square of the input."""

    @abstractmethod
    def absolute(self, x) -> Any:
        """Calculate the absolute value element-wise."""

    @abstractmethod
    def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Compute the arithmetic mean along the specified axis."""

    @abstractmethod
    def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the variance along the specified axis."""

    @abstractmethod
    def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the standard deviation along the specified axis."""

    @abstractmethod
    def min(self, a, axis=None, keepdims=False) -> Any:
        """Return minimum of an array or minimum along an axis."""

    @abstractmethod
    def max(self, a, axis=None, keepdims=False) -> Any:
        """Return maximum of an array or maximum along an axis."""

    @abstractmethod
    def argmin(self, a, axis=None) -> Any:
        """Return indices of the minimum values along the given axis."""

    @abstractmethod
    def argmax(self, a, axis=None) -> Any:
        """Return indices of the maximum values along the given axis."""

    @abstractmethod
    def clip(self, a, a_min, a_max) -> Any:
        """Clip (limit) the values in an array."""

    @abstractmethod
    def round(self, a, decimals=0) -> Any:
        """Round an array to the given number of decimals."""

    @abstractmethod
    def dot(self, a, b) -> Any:
        """Dot product of two arrays."""

    @abstractmethod
    def tensordot(self, a, b, axes=2) -> Any:
        """Compute tensor dot product along specified axes."""

    @abstractmethod
    def matmul(self, a, b) -> Any:
        """Matrix product of two arrays."""

    # Statistical functions
    @abstractmethod
    def normal(self, loc=0.0, scale=1.0, shape=None, dtype=None, key=None) -> Any:
        """Draw random samples from a normal (Gaussian) distribution."""

    @abstractmethod
    def uniform(self, minval=0.0, maxval=1.0, shape=None, dtype=None, key=None) -> Any:
        """Draw random samples from a uniform distribution."""

    def random_seed(self, seed=None) -> None:
        """Seed the random number generator."""
        self._not_implemented("random_seed")

    def random_split(self, key, num=2) -> Any:
        """Split a PRNG key into multiple keys."""
        self._not_implemented("random_split")

    # Linear algebra operations
    @abstractmethod
    def inv(self, a) -> Any:
        """Compute the inverse of a matrix."""

    @abstractmethod
    def eigh(self, a) -> tuple[Any, Any]:
        """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""

    @abstractmethod
    def cholesky(self, a) -> Any:
        """Cholesky decomposition."""

    @abstractmethod
    def solve(self, a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""

    # Differential operations
    @abstractmethod
    def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the gradient of `fun` with respect to positional arguments."""

    @abstractmethod
    def value_and_grad(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function that evaluates both fun and its gradient."""

    @abstractmethod
    def hessian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function to compute the Hessian of `fun` with respect to positional arguments."""

    @abstractmethod
    def jacobian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function to compute the Jacobian of `fun` with respect to positional arguments."""

    @abstractmethod
    def custom_jvp(self, fun: Callable, jvp: Optional[Callable] = None) -> Callable:
        """Specify a custom JVP rule for a function."""

    @abstractmethod
    def custom_vjp(
        self,
        fun: Callable,
        fwd: Optional[Callable] = None,
        bwd: Optional[Callable] = None,
    ) -> Callable:
        """Specify a custom VJP rule for a function."""

    # Array manipulation
    @abstractmethod
    def reshape(self, a, newshape) -> Any:
        """Gives a new shape to an array without changing its data."""

    @abstractmethod
    def transpose(self, a, axes=None) -> Any:
        """Permute the dimensions of an array."""

    @abstractmethod
    def concatenate(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along an existing axis."""

    @abstractmethod
    def stack(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along a new axis."""

    @abstractmethod
    def vstack(self, tup) -> Any:
        """Stack arrays in sequence vertically (row wise)."""

    @abstractmethod
    def hstack(self, tup) -> Any:
        """Stack arrays in sequence horizontally (column wise)."""

    @abstractmethod
    def where(self, condition, x=None, y=None) -> Any:
        """Return elements chosen from `x` or `y` depending on `condition`."""

    # Control flow operations
    @abstractmethod
    def scan(self, f, init, xs, length=None, reverse=False) -> Any:
        """Scan a function over leading array axes while carrying along state."""

    # PyTree operations
    @abstractmethod
    def tree_map(self, f: Callable[[T], U], tree: T, *rest: Any) -> U:
        """Map a function over a pytree."""

    @abstractmethod
    def tree_flatten(self, tree: T) -> tuple[list[Any], Any]:
        """Flatten a pytree into a list of leaves and a treedef."""

    @abstractmethod
    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        """Unflatten a list of leaves and a treedef into a pytree."""

    # Transformations
    @abstractmethod
    def jit(
        self, fun: Callable, static_argnums: Union[int, Sequence[int]] = ()
    ) -> Callable:
        """JIT-compile a function for faster execution."""

    @abstractmethod
    def vmap(self, fun: Callable, in_axes=0, out_axes=0) -> Callable:
        """Vectorize a function along the specified axes."""

    @abstractmethod
    def pmap(self, fun: Callable, axis_name=None, devices=None) -> Callable:
        """Parallel map over an axis."""

    @abstractmethod
    def checkpoint(self, fun: Callable, concrete: bool = False) -> Callable:
        """Checkpoint a function to save memory during backpropagation."""

    # Additional JAX transformations
    def cond(self, pred, true_fun, false_fun, *operands) -> Any:
        """Conditionally apply true_fun or false_fun based on the value of pred."""
        self._not_implemented("cond")

    def while_loop(self, cond_fun, body_fun, init_val) -> Any:
        """Apply body_fun repeatedly while cond_fun is true."""
        self._not_implemented("while_loop")

    def fori_loop(self, lower, upper, body_fun, init_val) -> Any:
        """Apply body_fun over the range from lower to upper."""
        self._not_implemented("fori_loop")

    def switch(self, index, branches, *operands) -> Any:
        """Apply branches[index] to operands."""
        self._not_implemented("switch")

    def device_put(self, x, device=None) -> Any:
        """Transfer x to the specified device."""
        self._not_implemented("device_put")

    def host_callback(self, callback, arg, *, result_shape=None, identity=None) -> Any:
        """Call a Python function on the host during computation."""
        self._not_implemented("host_callback")

    def xmap(
        self, fun, in_axes, out_axes, *, axis_resources=None, backend=None
    ) -> Callable:
        """Vectorize a function with named axes."""
        self._not_implemented("xmap")

    def pjit(
        self, fun, in_shardings, out_shardings, *, static_argnums=None
    ) -> Callable:
        """JIT with more flexible sharding."""
        self._not_implemented("pjit")

    # Other standard functions
    @abstractmethod
    def sign(self, x) -> Any:
        """Returns an element-wise indication of the sign of a number."""

    @abstractmethod
    def floor(self, x) -> Any:
        """Return the floor of the input, element-wise."""

    @abstractmethod
    def ceil(self, x) -> Any:
        """Return the ceiling of the input, element-wise."""

    # Helper methods
    def _not_implemented(self, func_name: str) -> Any:
        """Raise NotImplementedInBackend error."""
        raise NotImplementedInBackend(func_name, self.name)
