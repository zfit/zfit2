"""Tests for the enhanced backend system."""

from __future__ import annotations

import numpy as np
import pytest

import zfit2.backend as z
from zfit2.backend import numpy as znp
from zfit2.backend.context import host_callback, numpy_fallback, use_backend

# Skip tests if optional dependencies are not available
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def test_backend_interface():
    """Test the basic backend interface."""
    # Test numpy interface
    x = znp.array([1.0, 2.0, 3.0])
    y = znp.sin(x)
    np.testing.assert_allclose(y, np.sin(np.array([1.0, 2.0, 3.0])))

    # Test backend interface
    grad_fn = z.backend.grad(lambda x: znp.sum(x**2))
    grad = grad_fn(x)
    np.testing.assert_allclose(grad, 2 * np.array([1.0, 2.0, 3.0]))


def test_backend_switching():
    """Test switching between backends."""
    x = znp.array([1.0, 2.0, 3.0])

    # Default backend should be JAX if available, otherwise NumPy
    default_backend = z.get_backend().name
    expected_default = "JAX" if HAS_JAX else "NumPy"
    assert default_backend == expected_default

    # Switch to NumPy backend
    with use_backend("numpy"):
        assert z.get_backend().name == "NumPy"
        y_numpy = znp.mean(x)

    # Back to default backend
    assert z.get_backend().name == default_backend

    # Compare results
    y_default = znp.mean(x)
    np.testing.assert_allclose(y_numpy, y_default)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_enhancements():
    """Test JAX-specific enhancements."""
    # Ensure JAX backend is active
    with use_backend("jax"):
        # Test cond
        def f_cond(x):
            return z.backend.cond(x > 0, lambda x: x**2, lambda x: -x, x)

        assert f_cond(2.0) == 4.0
        assert f_cond(-2.0) == 2.0

        # Test while_loop
        def f_while(x):
            def cond_fun(val):
                return val < 10

            def body_fun(val):
                return val * 2

            return z.backend.while_loop(cond_fun, body_fun, x)

        assert f_while(1.0) == 16.0

        # Test fori_loop
        def f_fori(x):
            def body_fun(i, val):
                return val + i

            return z.backend.fori_loop(0, 5, body_fun, x)

        assert f_fori(0.0) == 10.0

        # Test device_put
        x = znp.array([1.0, 2.0, 3.0])
        x_dev = z.backend.device_put(x)
        assert isinstance(x_dev, jax.Array)

        # Test host_callback
        def callback_fn(x):
            return float(np.mean(x))

        def f_callback(x):
            return z.backend.host_callback(callback_fn, x)

        assert f_callback(jnp.array([1.0, 2.0, 3.0])) == 2.0


@pytest.mark.skipif(not HAS_SYMPY, reason="SymPy not installed")
def test_sympy_enhancements():
    """Test SymPy-specific enhancements."""
    # Switch to SymPy backend
    with use_backend("sympy"):
        # Create a symbolic expression
        x = sympy.Symbol("x")
        expr = znp.sin(x) ** 2 + znp.cos(x) ** 2

        # Test simplification
        simplified = z.backend.simplify(expr)
        assert simplified == 1

        # Test differentiation
        deriv = z.backend.diff(znp.sin(x), x)
        assert deriv == sympy.cos(x)

        # Test JIT compilation with optimization
        def f(x):
            return sympy.sin(x) ** 2 + sympy.cos(x) ** 2

        f_jit = z.backend.jit(f)
        result = f_jit(sympy.Symbol("x"))
        assert result == 1


def test_host_callback_decorator():
    """Test the host_callback decorator."""

    # Define a host function
    def host_fn(x):
        return float(np.mean(x))

    # Apply the decorator
    @host_callback(host_fn)
    def decorated_fn(x):
        return x

    # Test with NumPy array
    x = np.array([1.0, 2.0, 3.0])
    assert decorated_fn(x) == 2.0

    # Test with JAX array (if available)
    if HAS_JAX:
        with use_backend("jax"):
            x_jax = jnp.array([1.0, 2.0, 3.0])
            assert decorated_fn(x_jax) == 2.0


def test_numpy_fallback_decorator():
    """Test the numpy_fallback decorator."""

    # Define a function that fails in SymPy
    @numpy_fallback
    def complex_fn(x):
        # This will fail in SymPy but work in NumPy/JAX
        return znp.std(x)

    # Test with NumPy backend
    with use_backend("numpy"):
        x = znp.array([1.0, 2.0, 3.0])
        result_numpy = complex_fn(x)

    # Test with SymPy backend (if available)
    if HAS_SYMPY:
        with use_backend("sympy"):
            # Should automatically fall back to NumPy
            result_sympy = complex_fn(x)
            np.testing.assert_allclose(result_sympy, result_numpy)


def test_vectorization():
    """Test vectorization utilities."""
    # Create test data
    x = znp.array([[1.0, 2.0], [3.0, 4.0]])
    y = znp.array([[10.0, 20.0], [30.0, 40.0]])

    # Define a simple function to vectorize
    def add(a, b):
        return a + b

    # Use vmap to vectorize over the first dimension
    add_rows = z.vmap(add)
    result = add_rows(x, y)

    # Check the result
    expected = znp.array([[11.0, 22.0], [33.0, 44.0]])
    np.testing.assert_allclose(result, expected)


def test_optimization():
    """Test optimization utilities."""
    from zfit2.backend.optimize import minimize

    # Define a simple quadratic function to minimize
    def f(x):
        return znp.sum(x**2)

    # Minimize the function
    result = minimize(f, znp.array([1.0, 1.0]))

    # Check that the minimum is at [0, 0]
    np.testing.assert_allclose(result.x, np.zeros(2), atol=1e-5)


def test_backends_compatibility():
    """Test compatibility between different backends."""
    # Create test data
    x = np.array([1.0, 2.0, 3.0])

    # Define a function to test across backends
    def f(x):
        return znp.sin(x) + znp.cos(x)

    # Compute with NumPy backend
    with use_backend("numpy"):
        result_numpy = f(x)

    # Compute with JAX backend (if available)
    if HAS_JAX:
        with use_backend("jax"):
            result_jax = f(x)
            np.testing.assert_allclose(result_jax, result_numpy)

    # For SymPy, we can only test with scalar inputs
    if HAS_SYMPY:
        with use_backend("sympy"):
            scalar_x = 1.0
            result_sympy = float(f(scalar_x))
            scalar_numpy = float(np.sin(scalar_x) + np.cos(scalar_x))
            np.testing.assert_allclose(result_sympy, scalar_numpy)


def test_pytree_operations():
    """Test PyTree operations."""
    # Create a simple pytree
    pytree = {
        "a": znp.array([1.0, 2.0, 3.0]),
        "b": (znp.array([4.0, 5.0]), znp.array([6.0, 7.0])),
        "c": {"d": znp.array([8.0, 9.0, 10.0])},
    }

    # Test tree_map
    def square(x):
        return x**2

    squared_tree = z.backend.tree_map(square, pytree)

    # Check a few values
    np.testing.assert_allclose(squared_tree["a"], np.array([1.0, 4.0, 9.0]))
    np.testing.assert_allclose(squared_tree["b"][0], np.array([16.0, 25.0]))
    np.testing.assert_allclose(squared_tree["c"]["d"], np.array([64.0, 81.0, 100.0]))


if __name__ == "__main__":
    # Run tests
    test_backend_interface()
    test_backend_switching()

    if HAS_JAX:
        test_jax_enhancements()

    if HAS_SYMPY:
        test_sympy_enhancements()

    test_host_callback_decorator()
    test_numpy_fallback_decorator()
    test_vectorization()
    test_optimization()
    test_backends_compatibility()
    test_pytree_operations()

    print("All tests passed!")
