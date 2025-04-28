"""NumPy backend implementation."""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import numpy as np
from scipy import linalg, optimize, special

from .base import BackendBase

# Type variables for PyTree operations
T = TypeVar("T")
U = TypeVar("U")


class NumPyBackend(BackendBase):
    """Backend implementation using NumPy and SciPy."""

    def __init__(self):
        """Initialize the NumPy backend."""
        if not importlib.util.find_spec("scipy"):
            raise ImportError(
                "SciPy is not installed. Please install it with `pip install scipy`."
            )

        self._np = np
        self._special = special
        self._linalg = linalg
        self._optimize = optimize

        # Initialize random number generator
        self._rng = np.random.default_rng()

    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return "NumPy"

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the NumPy module."""
        try:
            return getattr(self._np, name)
        except AttributeError:
            try:
                return getattr(self._special, name)
            except AttributeError:
                try:
                    return getattr(self._linalg, name)
                except AttributeError:
                    self._not_implemented(name)

    # Core array creation functions
    def array(self, obj, dtype=None, copy=None, device=None) -> Any:
        """Create a NumPy array."""
        # In NumPy, 'device' is ignored
        return self._np.array(obj, dtype=dtype, copy=copy)

    def asarray(self, a, dtype=None, copy=None, device=None) -> Any:
        """Convert the input to an array."""
        # In NumPy, 'device' is ignored
        return self._np.asarray(a, dtype=dtype)

    def zeros(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with zeros."""
        # In NumPy, 'device' is ignored
        return self._np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with ones."""
        # In NumPy, 'device' is ignored
        return self._np.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with fill_value."""
        # In NumPy, 'device' is ignored
        return self._np.full(shape, fill_value, dtype=dtype)

    # Math operations
    def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Sum of array elements over given axes."""
        return self._np.sum(a, axis=axis, dtype=dtype, keepdims=keepdims)

    def exp(self, x) -> Any:
        """Calculate the exponential of all elements in the input array."""
        return self._np.exp(x)

    def log(self, x) -> Any:
        """Natural logarithm, element-wise."""
        return self._np.log(x)

    def sin(self, x) -> Any:
        """Sine, element-wise."""
        return self._np.sin(x)

    def cos(self, x) -> Any:
        """Cosine, element-wise."""
        return self._np.cos(x)

    def tan(self, x) -> Any:
        """Tangent, element-wise."""
        return self._np.tan(x)

    def arcsin(self, x) -> Any:
        """Inverse sine, element-wise."""
        return self._np.arcsin(x)

    def arccos(self, x) -> Any:
        """Inverse cosine, element-wise."""
        return self._np.arccos(x)

    def arctan(self, x) -> Any:
        """Inverse tangent, element-wise."""
        return self._np.arctan(x)

    def sinh(self, x) -> Any:
        """Hyperbolic sine, element-wise."""
        return self._np.sinh(x)

    def cosh(self, x) -> Any:
        """Hyperbolic cosine, element-wise."""
        return self._np.cosh(x)

    def tanh(self, x) -> Any:
        """Hyperbolic tangent, element-wise."""
        return self._np.tanh(x)

    def arcsinh(self, x) -> Any:
        """Inverse hyperbolic sine, element-wise."""
        return self._np.arcsinh(x)

    def arccosh(self, x) -> Any:
        """Inverse hyperbolic cosine, element-wise."""
        return self._np.arccosh(x)

    def arctanh(self, x) -> Any:
        """Inverse hyperbolic tangent, element-wise."""
        return self._np.arctanh(x)

    def power(self, x1, x2) -> Any:
        """First array elements raised to powers from second array, element-wise."""
        return self._np.power(x1, x2)

    def sqrt(self, x) -> Any:
        """Return the non-negative square-root of an array, element-wise."""
        return self._np.sqrt(x)

    def square(self, x) -> Any:
        """Return the element-wise square of the input."""
        return self._np.square(x)

    def absolute(self, x) -> Any:
        """Calculate the absolute value element-wise."""
        return self._np.absolute(x)

    def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Compute the arithmetic mean along the specified axis."""
        return self._np.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)

    def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the variance along the specified axis."""
        return self._np.var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

    def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Compute the standard deviation along the specified axis."""
        return self._np.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)

    def min(self, a, axis=None, keepdims=False) -> Any:
        """Return minimum of an array or minimum along an axis."""
        return self._np.min(a, axis=axis, keepdims=keepdims)

    def max(self, a, axis=None, keepdims=False) -> Any:
        """Return maximum of an array or maximum along an axis."""
        return self._np.max(a, axis=axis, keepdims=keepdims)

    def argmin(self, a, axis=None) -> Any:
        """Return indices of the minimum values along the given axis."""
        return self._np.argmin(a, axis=axis)

    def argmax(self, a, axis=None) -> Any:
        """Return indices of the maximum values along the given axis."""
        return self._np.argmax(a, axis=axis)

    def clip(self, a, a_min, a_max) -> Any:
        """Clip (limit) the values in an array."""
        return self._np.clip(a, a_min, a_max)

    def round(self, a, decimals=0) -> Any:
        """Round an array to the given number of decimals."""
        return self._np.round(a, decimals=decimals)

    def dot(self, a, b) -> Any:
        """Dot product of two arrays."""
        return self._np.dot(a, b)

    def tensordot(self, a, b, axes=2) -> Any:
        """Compute tensor dot product along specified axes."""
        return self._np.tensordot(a, b, axes=axes)

    def matmul(self, a, b) -> Any:
        """Matrix product of two arrays."""
        return self._np.matmul(a, b)

    # Statistical functions
    def normal(self, key=None, shape=None, dtype=None, loc=0.0, scale=1.0) -> Any:
        """Draw random samples from a normal distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.normal(loc=loc, scale=scale, size=shape).astype(dtype)

    def uniform(self, key=None, shape=None, dtype=None, minval=0.0, maxval=1.0) -> Any:
        """Draw random samples from a uniform distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.uniform(low=minval, high=maxval, size=shape).astype(dtype)

    def random_seed(self, seed=None) -> None:
        """Set the random seed for the random number generator.

        Args:
            seed: The seed to use. If None, a random seed will be used.
        """
        self._rng = np.random.default_rng(seed)

    def random_split(self, key, num=2) -> Any:
        """Split a PRNG key into multiple keys.

        In NumPy, we don't have explicit PRNG keys, so we create new random seeds.
        """
        if key is None:
            # Create a seed
            seed = np.random.randint(0, 2**31)
        else:
            # Use the provided key as seed
            seed = key

        # Create 'num' new seeds
        np.random.seed(seed)
        new_keys = np.random.randint(0, 2**31, size=num)

        return new_keys

    def choice(self, a, shape=None, replace=True, p=None, key=None) -> Any:
        """Random choice from an array.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        return self._rng.choice(a, size=shape, replace=replace, p=p)

    def permutation(self, x, key=None) -> Any:
        """Randomly permute a sequence, or return a permuted range.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        return self._rng.permutation(x)

    def shuffle(self, x, key=None) -> None:
        """Modify a sequence in-place by shuffling its contents.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        self._rng.shuffle(x)

    def exponential(self, scale=1.0, shape=None, dtype=None, key=None) -> Any:
        """Draw samples from an exponential distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.exponential(scale=scale, size=shape).astype(dtype)

    def poisson(self, lam=1.0, shape=None, dtype=None, key=None) -> Any:
        """Draw samples from a Poisson distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.poisson(lam=lam, size=shape).astype(dtype)

    def beta(self, a, b, shape=None, dtype=None, key=None) -> Any:
        """Draw samples from a Beta distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.beta(a=a, b=b, size=shape).astype(dtype)

    def gamma(self, shape, scale=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a Gamma distribution.

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if size is None:
            size = ()
        return self._rng.gamma(shape=shape, scale=scale, size=size).astype(dtype)

    def randint(self, low, high=None, shape=None, dtype=None, key=None) -> Any:
        """Return random integers from low (inclusive) to high (exclusive).

        For compatibility with JAX, we accept a 'key' parameter, but it's ignored.
        """
        if shape is None:
            shape = ()
        return self._rng.integers(low=low, high=high, size=shape, dtype=dtype)

    # Linear algebra operations
    def inv(self, a) -> Any:
        """Compute the inverse of a matrix."""
        return self._np.linalg.inv(a)

    def eigh(self, a) -> tuple[Any, Any]:
        """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        return self._np.linalg.eigh(a)

    def cholesky(self, a) -> Any:
        """Cholesky decomposition."""
        return self._np.linalg.cholesky(a)

    def solve(self, a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""
        return self._np.linalg.solve(a, b)

    # Differential operations
    def _finite_diff_gradient(self, func, x, eps=1e-8):
        """Compute the gradient of a function using finite differences."""
        x_array = np.asarray(x)
        grad = np.zeros_like(x_array, dtype=float)

        # Handle scalar case
        if np.isscalar(x) or x_array.ndim == 0:
            x_plus = x + eps
            x_minus = x - eps
            grad = (func(x_plus) - func(x_minus)) / (2 * eps)
            return grad

        for i in range(x_array.size):
            idx = np.unravel_index(i, x_array.shape)
            x_plus = x_array.copy()
            x_plus[idx] += eps
            x_minus = x_array.copy()
            x_minus[idx] -= eps

            # Use central difference method
            grad_val = (func(x_plus) - func(x_minus)) / (2 * eps)

            # Handle scalar output
            if np.isscalar(grad_val):
                grad[idx] = grad_val
            else:
                # For vector-valued functions, we need to compute the Jacobian
                # and extract the appropriate component
                grad[idx] = np.sum(grad_val)

        return grad

    def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function to compute the gradient of `fun` with respect to positional arguments."""
        if isinstance(argnums, int):
            argnums = (argnums,)

        def grad_fn(*args, **kwargs):
            arg = args[argnums[0]]
            if len(argnums) > 1:
                raise ValueError("Multiple argnums not implemented for NumPy backend")

            def partial_fun(x):
                new_args = list(args)
                new_args[argnums[0]] = x
                return fun(*new_args, **kwargs)

            return self._finite_diff_gradient(partial_fun, arg)

        return grad_fn

    def value_and_grad(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function that evaluates both fun and its gradient."""
        grad_fn = self.grad(fun, argnums)

        def value_and_grad_fn(*args, **kwargs):
            val = fun(*args, **kwargs)
            grad_val = grad_fn(*args, **kwargs)
            return val, grad_val

        return value_and_grad_fn

    def _finite_diff_hessian(self, func, x, eps=1e-5):
        """Compute the Hessian of a function using finite differences."""
        x_array = np.asarray(x)
        n = x_array.size
        hess = np.zeros((n, n))

        # Handle scalar case
        if np.isscalar(x) or x_array.ndim == 0:
            x_plus = x + eps
            x_center = x
            x_minus = x - eps
            hess = (func(x_plus) - 2 * func(x_center) + func(x_minus)) / (eps**2)
            return hess

        flat_x = x_array.ravel()

        for i in range(n):
            for j in range(i, n):
                idx_i = np.unravel_index(i, x_array.shape)
                idx_j = np.unravel_index(j, x_array.shape)

                x_pp = flat_x.copy()
                x_pp[i] += eps
                x_pp[j] += eps

                x_pm = flat_x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp = flat_x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm = flat_x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps

                # Reshape arrays back to original shape
                x_pp = x_pp.reshape(x_array.shape)
                x_pm = x_pm.reshape(x_array.shape)
                x_mp = x_mp.reshape(x_array.shape)
                x_mm = x_mm.reshape(x_array.shape)

                # Use central difference formula
                hess_val = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (
                    4 * eps * eps
                )

                # Handle scalar output
                if np.isscalar(hess_val):
                    hess[i, j] = hess_val
                    if i != j:
                        hess[j, i] = hess_val
                else:
                    # For vector-valued functions
                    hess_sum = np.sum(hess_val)
                    hess[i, j] = hess_sum
                    if i != j:
                        hess[j, i] = hess_sum

        return hess.reshape(x_array.shape + x_array.shape)

    def hessian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function to compute the Hessian of `fun` with respect to positional arguments."""
        if isinstance(argnums, int):
            argnums = (argnums,)

        def hessian_fn(*args, **kwargs):
            arg = args[argnums[0]]
            if len(argnums) > 1:
                raise ValueError("Multiple argnums not implemented for NumPy backend")

            def partial_fun(x):
                new_args = list(args)
                new_args[argnums[0]] = x
                return fun(*new_args, **kwargs)

            return self._finite_diff_hessian(partial_fun, arg)

        return hessian_fn

    def _finite_diff_jacobian(self, func, x, eps=1e-8):
        """Compute the Jacobian of a function using finite differences."""
        x_array = np.asarray(x)

        # Evaluate the function at the provided point
        f0 = func(x_array)
        f0_array = np.asarray(f0)

        # Handle scalar output
        if np.isscalar(f0) or f0_array.ndim == 0:
            return self._finite_diff_gradient(func, x_array, eps)

        # For vector output, we need a Jacobian matrix
        n = x_array.size  # Number of input variables
        m = f0_array.size  # Number of output variables

        jac = np.zeros((m, n))

        flat_x = x_array.ravel()

        for i in range(n):
            x_plus = flat_x.copy()
            x_plus[i] += eps
            x_minus = flat_x.copy()
            x_minus[i] -= eps

            # Reshape for function evaluation
            x_plus = x_plus.reshape(x_array.shape)
            x_minus = x_minus.reshape(x_array.shape)

            # Evaluate function at perturbed points
            f_plus = np.asarray(func(x_plus)).ravel()
            f_minus = np.asarray(func(x_minus)).ravel()

            # Compute derivative using central difference
            jac[:, i] = (f_plus - f_minus) / (2 * eps)

        # Reshape to match output dimensions
        output_shape = f0_array.shape + x_array.shape
        return jac.reshape(output_shape)

    def jacobian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function to compute the Jacobian of `fun` with respect to positional arguments."""
        if isinstance(argnums, int):
            argnums = (argnums,)

        def jacobian_fn(*args, **kwargs):
            arg = args[argnums[0]]
            if len(argnums) > 1:
                raise ValueError("Multiple argnums not implemented for NumPy backend")

            def partial_fun(x):
                new_args = list(args)
                new_args[argnums[0]] = x
                return fun(*new_args, **kwargs)

            return self._finite_diff_jacobian(partial_fun, arg)

        return jacobian_fn

    def custom_jvp(self, fun: Callable, jvp: Optional[Callable] = None) -> Callable:
        """Specify a custom JVP rule for a function.

        In NumPy, we simply return the function as-is with a warning.
        """
        if jvp is not None:
            import warnings

            warnings.warn(
                "Custom JVP rules are not supported in NumPy backend", UserWarning
            )
        return fun

    def custom_vjp(
        self,
        fun: Callable,
        fwd: Optional[Callable] = None,
        bwd: Optional[Callable] = None,
    ) -> Callable:
        """Specify a custom VJP rule for a function.

        In NumPy, we simply return the function as-is with a warning.
        """
        if fwd is not None or bwd is not None:
            import warnings

            warnings.warn(
                "Custom VJP rules are not supported in NumPy backend", UserWarning
            )
        return fun

    # Array manipulation
    def reshape(self, a, newshape) -> Any:
        """Gives a new shape to an array without changing its data."""
        return self._np.reshape(a, newshape)

    def transpose(self, a, axes=None) -> Any:
        """Permute the dimensions of an array."""
        return self._np.transpose(a, axes=axes)

    def concatenate(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along an existing axis."""
        return self._np.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along a new axis."""
        return self._np.stack(arrays, axis=axis)

    def vstack(self, tup) -> Any:
        """Stack arrays in sequence vertically (row wise)."""
        return self._np.vstack(tup)

    def hstack(self, tup) -> Any:
        """Stack arrays in sequence horizontally (column wise)."""
        return self._np.hstack(tup)

    def where(self, condition, x=None, y=None) -> Any:
        """Return elements chosen from `x` or `y` depending on `condition`."""
        return self._np.where(condition, x, y)

    # Control flow operations
    def scan(self, f, init, xs, length=None, reverse=False) -> Any:
        """Scan a function over leading array axes while carrying along state."""
        xs_array = np.asarray(xs)

        if length is not None:
            # Use only the first 'length' elements
            if isinstance(xs_array, np.ndarray) and xs_array.ndim > 0:
                xs_array = xs_array[:length]

        if reverse:
            if isinstance(xs_array, np.ndarray) and xs_array.ndim > 0:
                xs_array = xs_array[::-1]

        carry = init
        ys = []

        for x in xs_array:
            carry, y = f(carry, x)
            ys.append(y)

        if reverse:
            ys = ys[::-1]

        return carry, np.array(ys)

    # Additional control flow operations
    def cond(self, pred, true_fun, false_fun, *operands) -> Any:
        """Conditionally apply true_fun or false_fun based on the value of pred."""
        if pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)

    def while_loop(self, cond_fun, body_fun, init_val) -> Any:
        """Apply body_fun repeatedly while cond_fun is true."""
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val

    def fori_loop(self, lower, upper, body_fun, init_val) -> Any:
        """Apply body_fun over the range from lower to upper."""
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    def switch(self, index, branches, *operands) -> Any:
        """Apply branches[index] to operands."""
        if not (0 <= index < len(branches)):
            raise IndexError(
                f"Switch index {index} out of bounds for {len(branches)} branches"
            )
        return branches[index](*operands)

    def device_put(self, x, device=None) -> Any:
        """In NumPy, device placement is a no-op."""
        return np.asarray(x)

    def host_callback(self, callback, arg, *, result_shape=None, identity=None) -> Any:
        """In NumPy, host callback is just a function call."""
        return callback(arg)

    def xmap(
        self, fun, in_axes, out_axes, *, axis_resources=None, backend=None
    ) -> Callable:
        """For NumPy, simplify to vmap for basic cases."""
        import warnings

        warnings.warn(
            "xmap is not fully supported in NumPy backend, falling back to vmap",
            UserWarning,
        )

        # Convert named axes to positional axes for vmap
        if isinstance(in_axes, dict):
            # Simple conversion for the most common case
            in_axes_pos = 0
        else:
            in_axes_pos = in_axes

        return self.vmap(fun, in_axes=in_axes_pos)

    def pjit(
        self, fun, in_shardings, out_shardings, *, static_argnums=None
    ) -> Callable:
        """For NumPy, simplify to jit."""
        import warnings

        warnings.warn(
            "pjit is not supported in NumPy backend, falling back to jit", UserWarning
        )
        return self.jit(fun, static_argnums=static_argnums)

    # PyTree operations
    def tree_map(self, f: Callable[[T], U], tree: T, *rest: Any) -> U:
        """Map a function over a pytree.

        In NumPy, we implement a simplified version that works with lists, tuples, and dicts.
        """
        if isinstance(tree, (list, tuple)):
            tree_type = type(tree)
            if not rest:
                return tree_type(f(leaf) for leaf in tree)
            else:
                return tree_type(
                    f(leaf, *(r[i] for r in rest)) for i, leaf in enumerate(tree)
                )
        elif isinstance(tree, dict):
            if not rest:
                return {k: f(v) for k, v in tree.items()}
            else:
                return {k: f(v, *(r[k] for r in rest)) for k, v in tree.items()}
        # Leaf node
        elif not rest:
            return f(tree)
        else:
            return f(tree, *(r for r in rest))

    def tree_flatten(self, tree: T) -> tuple[list[Any], Any]:
        """Flatten a pytree into a list of leaves and a treedef.

        In NumPy, we implement a simplified version.
        """
        leaves = []
        treedef = []

        def flatten_helper(subtree, path):
            if isinstance(subtree, (list, tuple)):
                treedef.append((path, type(subtree), len(subtree)))
                for i, leaf in enumerate(subtree):
                    flatten_helper(leaf, path + [i])
            elif isinstance(subtree, dict):
                treedef.append((path, type(subtree), sorted(subtree.keys())))
                for k in sorted(subtree.keys()):
                    flatten_helper(subtree[k], path + [k])
            else:
                # Leaf node
                leaves.append(subtree)
                treedef.append((path, "leaf", None))

        flatten_helper(tree, [])

        return leaves, treedef

    def tree_unflatten(self, treedef: Any, leaves: list[Any]) -> Any:
        """Unflatten a list of leaves and a treedef into a pytree.

        In NumPy, we implement a simplified version.
        """
        leaf_idx = 0

        def unflatten_helper(path, treedef_list):
            path_dict = {}
            result = None

            for td_path, td_type, td_info in treedef_list:
                if td_path[: len(path)] == path and len(td_path) == len(path) + 1:
                    idx = td_path[-1]
                    if td_type == "leaf":
                        nonlocal leaf_idx
                        path_dict[idx] = leaves[leaf_idx]
                        leaf_idx += 1
                    else:
                        subtree_treedef = [
                            t for t in treedef_list if t[0][: len(td_path)] == td_path
                        ]
                        path_dict[idx] = unflatten_helper(td_path, subtree_treedef)

            # Find the entry in treedef that exactly matches this path
            for td_path, td_type, td_info in treedef_list:
                if td_path == path:
                    if td_type == list:
                        result = [path_dict.get(i) for i in range(td_info)]
                    elif td_type == tuple:
                        result = tuple(path_dict.get(i) for i in range(td_info))
                    elif td_type == dict:
                        result = {k: path_dict.get(k) for k in td_info}
                    elif td_type == "leaf":
                        result = path_dict.get(0)
                    break

            return result

        return unflatten_helper([], treedef)

    # Transformations
    def jit(
        self, fun: Callable, static_argnums: Union[int, Sequence[int]] = ()
    ) -> Callable:
        """JIT compilation for NumPy backend.

        For NumPy, this is just a simple wrapper that doesn't actually provide
        any speedup, but maintains API compatibility.

        Args:
            fun: The function to compile.
            static_argnums: Arguments that should be considered static (not used in NumPy).

        Returns:
            The same function, unchanged.
        """

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            return fun(*args, **kwargs)

        return wrapper

    def vmap(self, fun: Callable, in_axes=0, out_axes=0) -> Callable:
        """Vectorize a function along the specified axes.

        For NumPy, we implement a simple vectorization using numpy.vectorize.
        """
        # Simple case: single input, single output, default axes
        if in_axes == 0 and out_axes == 0:
            return np.vectorize(fun)

        # More complex cases
        def vectorized_fun(*args):
            # Convert in_axes to a tuple if it's not already
            if isinstance(in_axes, int) or in_axes is None:
                actual_in_axes = (in_axes,) * len(args)
            else:
                actual_in_axes = in_axes

            # Validate in_axes
            if len(actual_in_axes) != len(args):
                raise ValueError(
                    f"Length of in_axes ({len(actual_in_axes)}) must match number of arguments ({len(args)})"
                )

            # Determine the shape of the output by broadcasting the inputs along their respective axes
            broadcast_shape = None
            for arg, in_ax in zip(args, actual_in_axes, strict=False):
                if in_ax is not None:
                    arg_array = np.asarray(arg)
                    if arg_array.ndim > in_ax:
                        axis_shape = arg_array.shape[in_ax]
                        if broadcast_shape is None:
                            broadcast_shape = axis_shape
                        elif broadcast_shape != axis_shape:
                            raise ValueError(
                                f"Mismatched shapes for vectorization: {broadcast_shape} vs {axis_shape}"
                            )

            if broadcast_shape is None:
                # No vectorization needed
                return fun(*args)

            # Prepare the result array
            first_result = fun(
                *(
                    arg[0] if in_ax == 0 and np.asarray(arg).ndim > 0 else arg
                    for arg, in_ax in zip(args, actual_in_axes, strict=False)
                )
            )
            result_shape = (broadcast_shape,) + np.shape(first_result)
            result = np.zeros(result_shape, dtype=np.asarray(first_result).dtype)

            # Apply the function to each slice
            for i in range(broadcast_shape):
                # Extract the appropriate slice for each input
                sliced_args = []
                for arg, in_ax in zip(args, actual_in_axes, strict=False):
                    arg_array = np.asarray(arg)
                    if in_ax is not None and arg_array.ndim > in_ax:
                        # Take the i-th slice along the specified axis
                        sliced_arg = np.take(arg_array, i, axis=in_ax)
                    else:
                        # Use the entire argument
                        sliced_arg = arg
                    sliced_args.append(sliced_arg)

                # Apply the function and store the result
                result_i = fun(*sliced_args)
                # Handle different result shapes/types
                if i == 0:
                    if np.isscalar(result_i) or result_i.ndim == 0:
                        result[i] = result_i
                    else:
                        # Ensure result has the right shape for non-scalar outputs
                        result_shape = (broadcast_shape,) + np.shape(result_i)
                        result = np.zeros(
                            result_shape, dtype=np.asarray(result_i).dtype
                        )
                        result[i] = result_i
                else:
                    result[i] = result_i

            # Transpose the result if needed
            if out_axes != 0:
                # Determine the permutation of axes
                ndim = result.ndim
                perm = list(range(ndim))
                perm.remove(out_axes)
                perm.insert(0, out_axes)
                result = np.transpose(result, perm)

            return result

        return vectorized_fun

    def pmap(self, fun: Callable, axis_name=None, devices=None) -> Callable:
        """Parallel map over an axis.

        For NumPy, this is just a serial map.
        """
        import warnings

        warnings.warn(
            "pmap is not parallelized in NumPy backend, using serial execution",
            UserWarning,
        )

        def pmapped_fun(*args):
            # Assume first dimension is the one to map over
            results = []
            for slices in zip(*(arg for arg in args), strict=False):
                results.append(fun(*slices))
            return np.array(results)

        return pmapped_fun

    def checkpoint(self, fun: Callable, concrete: bool = False) -> Callable:
        """Checkpoint a function to save memory during backpropagation.

        For NumPy, this is just a pass-through.
        """
        return fun

    # Other standard functions
    def sign(self, x) -> Any:
        """Returns an element-wise indication of the sign of a number."""
        return self._np.sign(x)

    def floor(self, x) -> Any:
        """Return the floor of the input, element-wise."""
        return self._np.floor(x)

    def ceil(self, x) -> Any:
        """Return the ceiling of the input, element-wise."""
        return self._np.ceil(x)
