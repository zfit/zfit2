"""Just-in-time compilation for SymPy expressions."""

from __future__ import annotations

import functools
import importlib.util
import warnings
from collections.abc import Callable
from typing import Any, Optional


class SymPyJIT:
    """JIT compiler for SymPy expressions.

    This class provides functionality to compile SymPy expressions into
    efficient numerical functions using various backends (Numba, JAX, etc.).
    It also applies symbolic optimizations before compilation.
    """

    def __init__(self, optimization_level: int = 2, backend: Optional[str] = None):
        """Initialize the SymPy JIT compiler.

        Args:
            optimization_level: Level of symbolic optimization to apply (0-3).
                0: No optimization
                1: Basic optimizations (common subexpressions, simple patterns)
                2: Moderate optimizations (includes trigonometric identities)
                3: Aggressive optimizations (may be slower for simple expressions)
            backend: The compilation backend to use ('numba', 'jax', or None for auto-select).
        """
        self.optimization_level = optimization_level

        # Determine the backend
        if backend is None:
            # Auto-select based on available packages
            if importlib.util.find_spec("jax"):
                self.backend = "jax"
            elif importlib.util.find_spec("numba"):
                self.backend = "numba"
            else:
                self.backend = "numpy"
        else:
            self.backend = backend

        # Initialize the SymPy module
        if not importlib.util.find_spec("sympy"):
            raise ImportError(
                "SymPy is not installed. Please install it with `pip install sympy`."
            )

        import sympy

        self._sympy = sympy

        # Cache for compiled functions
        self._cache: dict[Any, Callable] = {}

    def _optimize_expression(self, expr) -> Any:
        """Apply symbolic optimizations to an expression.

        Args:
            expr: The SymPy expression to optimize.

        Returns:
            An optimized SymPy expression.
        """
        import sympy

        # Skip optimization if level is 0
        if self.optimization_level == 0:
            return expr

        # Apply common subexpression elimination
        expr = sympy.cse(expr, optimizations="basic")

        # Basic optimizations (level 1+)
        if self.optimization_level >= 1:
            # Simplify the expression
            expr = sympy.simplify(expr)

            # Combine similar terms
            if hasattr(expr, "expand"):
                expr = expr.expand().collect(sympy.Symbol("x"))

        # Moderate optimizations (level 2+)
        if self.optimization_level >= 2:
            # Apply trigonometric identities
            expr = sympy.trigsimp(expr)

            # Apply special function identities
            expr = sympy.expand_func(expr)

            # Rewrite expressions in a more efficient form
            expr = expr.rewrite("exp")

            # Constant folding
            expr = sympy.together(expr)

        # Aggressive optimizations (level 3)
        if self.optimization_level >= 3:
            # Hypergeometric function simplification
            expr = sympy.hyperexpand(expr)

            # More advanced algebraic simplifications
            expr = sympy.cancel(expr)

            # Combinatorial simplifications
            expr = sympy.combsimp(expr)

            # Use polynomial expansion and collection again
            if hasattr(expr, "expand"):
                expr = expr.expand()

                # Collect terms by symbols
                symbols = list(expr.free_symbols)
                if symbols:
                    for symbol in symbols:
                        expr = expr.collect(symbol)

        return expr

    def _compile_with_numpy(self, expr, symbols) -> Callable:
        """Compile a SymPy expression using NumPy.

        Args:
            expr: The SymPy expression to compile.
            symbols: The symbols in the expression.

        Returns:
            A compiled function that takes the symbol values as arguments.
        """
        import numpy as np
        import sympy

        # Create a lambda function
        lambda_func = sympy.lambdify(symbols, expr, modules="numpy")

        # Create a wrapper that handles NumPy arrays consistently
        @functools.wraps(lambda_func)
        def wrapper(*args):
            # Convert all arguments to NumPy arrays
            args_np = [np.asarray(arg) for arg in args]

            # Compute the result
            result = lambda_func(*args_np)

            # Ensure the result is a NumPy array
            if result is not None:
                result = np.asarray(result)

            return result

        return wrapper

    def _compile_with_numba(self, expr, symbols) -> Callable:
        """Compile a SymPy expression using Numba.

        Args:
            expr: The SymPy expression to compile.
            symbols: The symbols in the expression.

        Returns:
            A compiled function that takes the symbol values as arguments.
        """
        import numpy as np
        import sympy

        # First create a NumPy lambda function
        numpy_func = sympy.lambdify(symbols, expr, modules="numpy")

        # Import numba
        try:
            import numba
        except ImportError:
            warnings.warn("Numba not available, falling back to NumPy")
            return self._compile_with_numpy(expr, symbols)

        # Define a Numba function
        @numba.jit(nopython=True)
        def numba_func(*args):
            # Handle different argument shapes
            scalar_result = all(np.isscalar(arg) for arg in args)
            if scalar_result:
                return numpy_func(*args)
            else:
                # For array inputs, we need to loop manually
                # This is because numba's nopython mode can't directly use sympy's lambdify
                args_shape = np.broadcast_shapes(
                    *(np.asarray(arg).shape for arg in args)
                )
                result = np.zeros(args_shape, dtype=np.float64)

                # Create flat iterators
                args_iter = [np.nditer(arg) for arg in args]

                # Iterate over elements
                for idx in np.ndindex(args_shape):
                    # Get scalar values at this index
                    scalar_args = [next(arg_iter) for arg_iter in args_iter]

                    # Compute the result for these scalar values
                    result[idx] = numpy_func(*scalar_args)

                    # Reset iterators if needed
                    for arg_iter in args_iter:
                        arg_iter.reset()

                return result

        return numba_func

    def _compile_with_jax(self, expr, symbols) -> Callable:
        """Compile a SymPy expression using JAX.

        Args:
            expr: The SymPy expression to compile.
            symbols: The symbols in the expression.

        Returns:
            A compiled function that takes the symbol values as arguments.
        """
        import sympy

        # Import JAX
        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            warnings.warn("JAX not available, falling back to NumPy")
            return self._compile_with_numpy(expr, symbols)

        # Create a lambda function that uses JAX's numpy
        lambda_func = sympy.lambdify(
            symbols, expr, modules=[{"ImmutableMatrix": jnp.array}, "numpy"]
        )

        # JIT-compile the function
        jitted_func = jax.jit(lambda_func)

        # Create a wrapper to ensure JAX arrays are used
        @functools.wraps(jitted_func)
        def wrapper(*args):
            # Convert all arguments to JAX arrays
            args_jax = [jnp.asarray(arg) for arg in args]

            # Compute the result
            result = jitted_func(*args_jax)

            return result

        return wrapper

    def compile(self, expr, symbols) -> Callable:
        """Compile a SymPy expression into an efficient numerical function.

        Args:
            expr: The SymPy expression to compile.
            symbols: The symbols in the expression (variables to substitute).

        Returns:
            A compiled function that takes the symbol values as arguments.
        """
        # Optimize the expression
        opt_expr = self._optimize_expression(expr)

        # Create a cache key based on the optimized expression and symbols
        cache_key = (opt_expr, tuple(symbols))

        # Check if the function is already in the cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compile based on the selected backend
        if self.backend == "jax":
            compiled_func = self._compile_with_jax(opt_expr, symbols)
        elif self.backend == "numba":
            compiled_func = self._compile_with_numba(opt_expr, symbols)
        else:
            # Default to NumPy
            compiled_func = self._compile_with_numpy(opt_expr, symbols)

        # Cache the compiled function
        self._cache[cache_key] = compiled_func

        return compiled_func

    def jit(self, func: Callable) -> Callable:
        """JIT-compile a Python function that contains SymPy expressions.

        This function analyzer the input function to identify SymPy expressions,
        optimizes them, and returns a compiled version of the function.

        Args:
            func: The function to compile.

        Returns:
            A compiled version of the function.
        """
        import inspect

        import sympy

        # Get the function's source code
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            warnings.warn(
                f"Couldn't get source for {func.__name__}, returning original function"
            )
            return func

        # Check if function uses SymPy
        if "sympy" not in source:
            warnings.warn(
                f"No SymPy usage detected in {func.__name__}, returning original function"
            )
            return func

        # Get function signature
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # First call the original function to identify SymPy expressions
            result = func(*args, **kwargs)

            # If the result is a SymPy expression, compile and evaluate it
            if isinstance(result, sympy.Expr):
                # Get the symbols (free variables) in the expression
                symbols = list(result.free_symbols)

                # We need to map these symbols to the function arguments
                # This is a simplified approach; in a real implementation,
                # you'd need to properly track symbol-to-argument mapping

                # Bind arguments to the function's parameters
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract values for symbols
                symbol_values = []
                for symbol in symbols:
                    # Assume symbol names match parameter names
                    # This is a simplification and might not work in all cases
                    symbol_name = symbol.name
                    if symbol_name in bound_args.arguments:
                        symbol_values.append(bound_args.arguments[symbol_name])
                    else:
                        # If symbol doesn't match a parameter, use a default value
                        # This is just a fallback and might cause incorrect results
                        symbol_values.append(1.0)
                        warnings.warn(
                            f"Symbol {symbol_name} not found in function arguments"
                        )

                # Compile the expression
                compiled_func = self.compile(result, symbols)

                # Evaluate with the symbol values
                return compiled_func(*symbol_values)

            # If the result is a list or tuple containing SymPy expressions, compile each one
            elif isinstance(result, (list, tuple)) and any(
                isinstance(item, sympy.Expr) for item in result
            ):
                compiled_results = []

                for item in result:
                    if isinstance(item, sympy.Expr):
                        # Get the symbols in the expression
                        symbols = list(item.free_symbols)

                        # Bind arguments to the function's parameters
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        # Extract values for symbols
                        symbol_values = []
                        for symbol in symbols:
                            symbol_name = symbol.name
                            if symbol_name in bound_args.arguments:
                                symbol_values.append(bound_args.arguments[symbol_name])
                            else:
                                symbol_values.append(1.0)
                                warnings.warn(
                                    f"Symbol {symbol_name} not found in function arguments"
                                )

                        # Compile and evaluate
                        compiled_func = self.compile(item, symbols)
                        compiled_results.append(compiled_func(*symbol_values))
                    else:
                        compiled_results.append(item)

                # Return the same type as the original result
                return type(result)(compiled_results)

            else:
                # If not a SymPy expression or container of them, return as is
                return result

        return wrapper
