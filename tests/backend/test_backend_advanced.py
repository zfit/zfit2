"""Tests for the enhanced backend system."""

from __future__ import annotations

import numpy as np
import pytest

import zfit2.backend as zb
from zfit2.backend import backend, numpy
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
    x = numpy.array([1.0, 2.0, 3.0])
    y = numpy.sin(x)
    np.testing.assert_allclose(y, np.sin(np.array([1.0, 2.0, 3.0])))

    # Test backend interface
    grad_fn = backend.grad(lambda x: numpy.sum(x**2))
    grad = grad_fn(x)
    np.testing.assert_allclose(grad, 2 * np.array([1.0, 2.0, 3.0]))


def test_backend_switching():
    """Test switching between backends."""
    x = numpy.array([1.0, 2.0, 3.0])

    # Default backend should be JAX if available, otherwise NumPy
    default_backend = zb.get_backend().name
    expected_default = "JAX" if HAS_JAX else "NumPy"
    assert default_backend == expected_default

    # Switch to NumPy backend
    with use_backend("numpy"):
        assert zb.get_backend().name == "NumPy"
        y_numpy = numpy.mean(x)

    # Back to default backend
    assert zb.get_backend().name == default_backend

    # Compare results
    y_default = numpy.mean(x)
    np.testing.assert_allclose(y_numpy, y_default)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_enhancements():
    """Test JAX-specific enhancements."""
    # Ensure JAX backend is active
    with use_backend("jax"):
        # Test cond
        def f_cond(x):
            return backend.cond(x > 0, lambda x: x**2, lambda x: -x, x)

        assert f_cond(2.0) == 4.0
        assert f_cond(-2.0) == 2.0

        # Test while_loop
        def f_while(x):
            def cond_fun(val):
                return val < 10

            def body_fun(val):
                return val * 2

            return backend.while_loop(cond_fun, body_fun, x)

        assert f_while(1.0) == 16.0

        # Test fori_loop
        def f_fori(x):
            def body_fun(i, val):
                return val + i

            return backend.fori_loop(0, 5, body_fun, x)

        assert f_fori(0.0) == 10.0

        # Test device_put
        x = numpy.array([1.0, 2.0, 3.0])
        x_dev = backend.device_put(x)
        assert isinstance(x_dev, jax.Array)

        # Test host_callback
        def callback_fn(x):
            return float(np.mean(x))

        def f_callback(x):
            return backend.host_callback(callback_fn, x)

        assert f_callback(jnp.array([1.0, 2.0, 3.0])) == 2.0


@pytest.mark.skipif(not HAS_SYMPY, reason="SymPy not installed")
def test_sympy_enhancements():
    """Test SymPy-specific enhancements."""
    # Switch to SymPy backend
    with use_backend("sympy"):
        # Create a symbolic expression
        x = sympy.Symbol("x")
        expr = numpy.sin(x) ** 2 + numpy.cos(x) ** 2

        # Test simplification
        simplified = backend.simplify(expr)
        assert simplified == 1

        # Test differentiation
        deriv = backend.diff(numpy.sin(x), x)
        assert deriv == sympy.cos(x)

        # Test JIT compilation with optimization
        def f(x):
            return sympy.sin(x) ** 2 + sympy.cos(x) ** 2

        f_jit = backend.jit(f)
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
        return numpy.std(x)

    # Test with NumPy backend
    with use_backend("numpy"):
        x = numpy.array([1.0, 2.0, 3.0])
        result_numpy = complex_fn(x)

    # Test with SymPy backend (if available)
    if HAS_SYMPY:
        with use_backend("sympy"):
            # Should automatically fall back to NumPy
            result_sympy = complex_fn(x)
            np.testing.assert_allclose(result_sympy, result_numpy)


def test_backends_compatibility():
    """Test compatibility between different backends."""
    # Create test data
    x = np.array([1.0, 2.0, 3.0])

    # Define a function to test across backends
    def f(x):
        return numpy.sin(x) + numpy.cos(x)

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
    test_backends_compatibility()

    print("All tests passed!")
