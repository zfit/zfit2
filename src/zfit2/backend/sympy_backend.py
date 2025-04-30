"""SymPy backend implementation."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import numpy as np

from .base import BackendBase
from .sympy_jit import SymPyJIT

# Type variables for PyTree operations
T = TypeVar("T")
U = TypeVar("U")


class SymPyBackend(BackendBase):
    """Backend implementation using SymPy."""

    def __init__(self):
        """Initialize the SymPy backend."""
        if not importlib.util.find_spec("sympy"):
            raise ImportError(
                "SymPy is not installed. Please install it with `pip install sympy`."
            )

        import sympy
        import sympy.matrices

        self._sympy = sympy
        self._np = np

        # Initialize the JIT compiler
        self._jit_compiler = SymPyJIT()

        # Try to import JAX for optimized numerical evaluation
        self._has_jax = importlib.util.find_spec("jax") is not None
        if self._has_jax:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp

    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return "SymPy"

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the SymPy module."""
        try:
            return getattr(self._sympy, name)
        except AttributeError:
            self._not_implemented(name)

    # Core array creation functions
    def array(self, obj, dtype=None, copy=None, device=None) -> Any:
        """Create a SymPy array."""
        if isinstance(obj, (list, tuple, np.ndarray)):
            import sympy

            # try:
            #     return sympy.Matrix(obj)
            # except:
            return sympy.Array(obj)
        return obj

    def asarray(self, a, dtype=None, copy=None, device=None) -> Any:
        """Convert the input to a SymPy array."""
        return self.array(a)

    def zeros(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with zeros."""
        import sympy

        if isinstance(shape, (tuple, list)):
            if len(shape) == 1:
                return sympy.zeros(shape[0], 1)
            elif len(shape) == 2:
                return sympy.zeros(shape[0], shape[1])
            else:
                raise ValueError(
                    f"SymPy only supports 1D and 2D arrays, got shape {shape}"
                )
        else:
            return sympy.zeros(shape, 1)

    def ones(self, shape, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with ones."""
        import sympy

        if isinstance(shape, (tuple, list)):
            if len(shape) == 1:
                return sympy.ones(shape[0], 1)
            elif len(shape) == 2:
                return sympy.ones(shape[0], shape[1])
            else:
                raise ValueError(
                    f"SymPy only supports 1D and 2D arrays, got shape {shape}"
                )
        else:
            return sympy.ones(shape, 1)

    def full(self, shape, fill_value, dtype=None, device=None) -> Any:
        """Return a new array of given shape and type, filled with fill_value."""
        arr = self.ones(shape)
        return arr * fill_value

    # Math operations
    def sum(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Sum of array elements."""
        import sympy

        if hasattr(a, "sum"):
            return a.sum()
        else:
            return sympy.Sum(a)

    def exp(self, x) -> Any:
        """Calculate the exponential of input."""
        return self._sympy.exp(x)

    def log(self, x) -> Any:
        """Natural logarithm."""
        return self._sympy.log(x)

    def sin(self, x) -> Any:
        """Sine function."""
        return self._sympy.sin(x)

    def cos(self, x) -> Any:
        """Cosine function."""
        return self._sympy.cos(x)

    def tan(self, x) -> Any:
        """Tangent function."""
        return self._sympy.tan(x)

    def arcsin(self, x) -> Any:
        """Inverse sine function."""
        return self._sympy.asin(x)

    def arccos(self, x) -> Any:
        """Inverse cosine function."""
        return self._sympy.acos(x)

    def arctan(self, x) -> Any:
        """Inverse tangent function."""
        return self._sympy.atan(x)

    def sinh(self, x) -> Any:
        """Hyperbolic sine function."""
        return self._sympy.sinh(x)

    def cosh(self, x) -> Any:
        """Hyperbolic cosine function."""
        return self._sympy.cosh(x)

    def tanh(self, x) -> Any:
        """Hyperbolic tangent function."""
        return self._sympy.tanh(x)

    def arcsinh(self, x) -> Any:
        """Inverse hyperbolic sine function."""
        return self._sympy.asinh(x)

    def arccosh(self, x) -> Any:
        """Inverse hyperbolic cosine function."""
        return self._sympy.acosh(x)

    def arctanh(self, x) -> Any:
        """Inverse hyperbolic tangent function."""
        return self._sympy.atanh(x)

    def power(self, x1, x2) -> Any:
        """Power function."""
        return x1**x2

    def sqrt(self, x) -> Any:
        """Square root function."""
        return self._sympy.sqrt(x)

    def square(self, x) -> Any:
        """Square function."""
        return x**2

    def absolute(self, x) -> Any:
        """Absolute value function."""
        return self._sympy.Abs(x)

    def mean(self, a, axis=None, dtype=None, keepdims=False) -> Any:
        """Mean of array elements."""
        self._not_implemented("mean")

    def var(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Variance of array elements."""
        self._not_implemented("var")

    def std(self, a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
        """Standard deviation of array elements."""
        self._not_implemented("std")

    def min(self, a, axis=None, keepdims=False) -> Any:
        """Minimum of array elements."""
        self._not_implemented("min")

    def max(self, a, axis=None, keepdims=False) -> Any:
        """Maximum of array elements."""
        self._not_implemented("max")

    def argmin(self, a, axis=None) -> Any:
        """Indices of minimum array elements."""
        self._not_implemented("argmin")

    def argmax(self, a, axis=None) -> Any:
        """Indices of maximum array elements."""
        self._not_implemented("argmax")

    def clip(self, a, a_min, a_max) -> Any:
        """Clip array elements."""
        self._not_implemented("clip")

    def round(self, a, decimals=0) -> Any:
        """Round array elements."""
        self._not_implemented("round")

    def dot(self, a, b) -> Any:
        """Dot product of arrays."""
        if hasattr(a, "dot"):
            return a.dot(b)
        self._not_implemented("dot")

    def tensordot(self, a, b, axes=2) -> Any:
        """Tensor dot product of arrays."""
        self._not_implemented("tensordot")

    def matmul(self, a, b) -> Any:
        """Matrix product of arrays."""
        if hasattr(a, "__mul__"):
            return a * b
        self._not_implemented("matmul")

    # Statistical functions
    def normal(self, key=None, shape=None, dtype=None, loc=0.0, scale=1.0) -> Any:
        """Symbolic normal distribution."""
        self._not_implemented("normal")

    def uniform(self, key=None, shape=None, dtype=None, minval=0.0, maxval=1.0) -> Any:
        """Symbolic uniform distribution."""
        self._not_implemented("uniform")

    def random_split(self, key, num=2) -> Any:
        """Split a PRNG key (not applicable for SymPy)."""
        self._not_implemented("random_split")

    # Linear algebra operations
    def inv(self, a) -> Any:
        """Compute the inverse of a matrix."""
        if hasattr(a, "inv"):
            return a.inv()
        self._not_implemented("inv")

    def eigh(self, a) -> tuple[Any, Any]:
        """Return eigenvalues and eigenvectors of a Hermitian or symmetric matrix."""
        if hasattr(a, "eigenvals") and hasattr(a, "eigenvects"):
            vals = a.eigenvals()
            vects = a.eigenvects()
            return list(vals.keys()), vects
        self._not_implemented("eigh")

    def cholesky(self, a) -> Any:
        """Cholesky decomposition."""
        self._not_implemented("cholesky")

    def solve(self, a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""
        if hasattr(a, "solve"):
            return a.solve(b)
        self._not_implemented("solve")

    # Differential operations
    def grad(self, fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
        """Return a function that computes the gradient of fun with respect to positional argument."""

        def grad_fn(*args, **kwargs):
            import sympy

            # Convert argnums to a tuple
            argnums_seq = argnums if isinstance(argnums, (list, tuple)) else (argnums,)

            # For each argnums, compute the gradient
            result = []
            for argnum in argnums_seq:
                arg = args[argnum]
                symbols = []

                # Create sympy symbols for the argument
                if isinstance(arg, (int, float, complex)):
                    x = sympy.symbols("x")
                    symbols.append(x)
                    new_args = list(args)
                    new_args[argnum] = x

                    # Compute the function with the symbolic argument
                    expr = fun(*new_args, **kwargs)

                    # Compute the derivative
                    deriv = sympy.diff(expr, x)

                    # Substitute back the original value
                    result.append(deriv.subs(x, arg))
                elif hasattr(arg, "free_symbols") and len(arg.free_symbols) > 0:
                    # If arg is already a symbolic expression
                    new_args = list(args)
                    expr = fun(*new_args, **kwargs)
                    deriv = sympy.diff(expr, arg)
                    result.append(deriv)
                else:
                    # For arrays, we'd need to create a symbol for each element
                    self._not_implemented("grad for arrays")

            if len(result) == 1:
                return result[0]
            return result

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

    def hessian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function that computes the Hessian of fun with respect to positional argument."""

        def hessian_fn(*args, **kwargs):
            import sympy

            # Convert argnums to a tuple
            argnums_seq = argnums if isinstance(argnums, (list, tuple)) else (argnums,)

            # For each argnums, compute the Hessian
            result = []
            for argnum in argnums_seq:
                arg = args[argnum]

                # Create sympy symbols for the argument
                if isinstance(arg, (int, float, complex)):
                    x = sympy.symbols("x")
                    new_args = list(args)
                    new_args[argnum] = x

                    # Compute the function with the symbolic argument
                    expr = fun(*new_args, **kwargs)

                    # Compute the second derivative
                    hess = sympy.diff(expr, x, 2)

                    # Substitute back the original value
                    result.append(hess.subs(x, arg))
                elif hasattr(arg, "free_symbols") and len(arg.free_symbols) > 0:
                    # If arg is already a symbolic expression
                    new_args = list(args)
                    expr = fun(*new_args, **kwargs)
                    hess = sympy.diff(expr, arg, 2)
                    result.append(hess)
                else:
                    # For arrays, we need to create a symbol for each element
                    self._not_implemented("hessian for arrays")

            if len(result) == 1:
                return result[0]
            return result

        return hessian_fn

    def jacobian(
        self, fun: Callable, argnums: Union[int, Sequence[int]] = 0
    ) -> Callable:
        """Return a function that computes the Jacobian of fun with respect to positional argument."""

        def jacobian_fn(*args, **kwargs):
            import sympy

            # Convert argnums to a tuple
            argnums_seq = argnums if isinstance(argnums, (list, tuple)) else (argnums,)

            # For each argnums, compute the Jacobian
            result = []
            for argnum in argnums_seq:
                arg = args[argnum]

                # For now, we only support scalar inputs and outputs
                if isinstance(arg, (int, float, complex)):
                    x = sympy.symbols("x")
                    new_args = list(args)
                    new_args[argnum] = x

                    # Compute the function with the symbolic argument
                    expr = fun(*new_args, **kwargs)

                    # Compute the derivative
                    jac = sympy.diff(expr, x)

                    # Substitute back the original value
                    result.append(jac.subs(x, arg))
                elif hasattr(arg, "free_symbols") and len(arg.free_symbols) > 0:
                    # If arg is already a symbolic expression
                    new_args = list(args)
                    expr = fun(*new_args, **kwargs)

                    # For vector-valued functions, compute Jacobian matrix
                    if hasattr(expr, "__iter__"):
                        jac_matrix = sympy.Matrix(
                            [[sympy.diff(f, var) for var in arg] for f in expr]
                        )
                        result.append(jac_matrix)
                    else:
                        # For scalar functions, compute gradient
                        jac = sympy.diff(expr, arg)
                        result.append(jac)
                else:
                    # For arrays or more complex inputs, we need more sophisticated handling
                    self._not_implemented("jacobian for arrays")

            if len(result) == 1:
                return result[0]
            return result

        return jacobian_fn

    def custom_jvp(self, fun: Callable, jvp: Optional[Callable] = None) -> Callable:
        """Specify a custom JVP rule for a function."""
        # In SymPy, we could implement this by customizing the differentiation rules,
        # but for now, we'll just return the original function.
        return fun

    def custom_vjp(
        self,
        fun: Callable,
        fwd: Optional[Callable] = None,
        bwd: Optional[Callable] = None,
    ) -> Callable:
        """Specify a custom VJP rule for a function."""
        # In SymPy, similar to custom_jvp, we'll just return the original function.
        return fun

    # Array manipulation
    def reshape(self, a, newshape) -> Any:
        """Gives a new shape to an array without changing its data."""
        self._not_implemented("reshape")

    def transpose(self, a, axes=None) -> Any:
        """Permute the dimensions of an array."""
        if hasattr(a, "transpose"):
            return a.transpose()
        self._not_implemented("transpose")

    def concatenate(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along an existing axis."""
        self._not_implemented("concatenate")

    def stack(self, arrays, axis=0) -> Any:
        """Join a sequence of arrays along a new axis."""
        self._not_implemented("stack")

    def vstack(self, tup) -> Any:
        """Stack arrays in sequence vertically (row wise)."""
        self._not_implemented("vstack")

    def hstack(self, tup) -> Any:
        """Stack arrays in sequence horizontally (column wise)."""
        self._not_implemented("hstack")

    def where(self, condition, x=None, y=None) -> Any:
        """Return elements chosen from `x` or `y` depending on `condition`."""
        self._not_implemented("where")

    # Control flow operations
    def scan(self, f, init, xs, length=None, reverse=False) -> Any:
        """Scan a function over leading array axes while carrying along state."""
        self._not_implemented("scan")

    # PyTree operations
    def tree_map(self, f: Callable[[T], U], tree: T, *rest: Any) -> U:
        """Map a function over a pytree."""
        # For SymPy, we'll implement a simplified version
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
        """Flatten a pytree into a list of leaves and a treedef."""
        # Simplified implementation for SymPy
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
        """Unflatten a list of leaves and a treedef into a pytree."""
        # Simplified implementation for SymPy
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
        """JIT-compile a function using SymPy optimization.

        Args:
            fun: The function to compile.
            static_argnums: Arguments that should be considered static (not used in SymPy).

        Returns:
            A compiled version of the function.
        """
        return self._jit_compiler.jit(fun)

    def vmap(self, fun: Callable, in_axes=0, out_axes=0) -> Callable:
        """Vectorize a function along the specified axes."""

        # For SymPy, we'll just return a function that raises NotImplementedInBackend
        def vectorized_fun(*args, **kwargs):
            self._not_implemented("vmap")

        return vectorized_fun

    def pmap(self, fun: Callable, axis_name=None, devices=None) -> Callable:
        """Parallel map over an axis."""

        # For SymPy, we'll just return a function that raises NotImplementedInBackend
        def pmapped_fun(*args, **kwargs):
            self._not_implemented("pmap")

        return pmapped_fun

    def checkpoint(self, fun: Callable, concrete: bool = False) -> Callable:
        """Checkpoint a function to save memory during backpropagation."""
        # For SymPy, we'll just return the original function
        return fun

    # Additional control flow operations
    def cond(self, pred, true_fun, false_fun, *operands) -> Any:
        """Conditionally apply true_fun or false_fun based on the value of pred."""
        # In SymPy, we can implement a symbolic conditional
        import sympy

        if isinstance(pred, (sympy.Basic, sympy.Expr)) and hasattr(pred, "is_Boolean"):
            # Symbolic predicate
            true_result = true_fun(*operands)
            false_result = false_fun(*operands)
            return sympy.Piecewise((true_result, pred), (false_result, True))
        # Non-symbolic predicate
        elif pred:
            return true_fun(*operands)
        else:
            return false_fun(*operands)

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
        """In SymPy, device placement is a no-op."""
        return x

    def host_callback(self, callback, arg, *, result_shape=None, identity=None) -> Any:
        """Call a Python function during symbolic execution."""
        # Just execute the callback directly
        return callback(arg)

    # Other standard functions
    def sign(self, x) -> Any:
        """Returns an element-wise indication of the sign of a number."""
        return self._sympy.sign(x)

    def floor(self, x) -> Any:
        """Return the floor of the input, element-wise."""
        return self._sympy.floor(x)

    def ceil(self, x) -> Any:
        """Return the ceiling of the input, element-wise."""
        return self._sympy.ceiling(x)
