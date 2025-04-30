"""Tests for the context managers and decorators in the backend module."""

from __future__ import annotations

import numpy as np
import pytest

import zfit2.backend as zb
from zfit2.backend import numpy as znp
from zfit2.backend.context import host_callback, numpy_fallback, use_backend


def test_use_backend_context_manager():
    """Test the use_backend context manager."""
    # Get initial backend
    initial_backend = zb.get_backend().name

    # Use numpy backend
    with use_backend("numpy"):
        assert zb.get_backend().name == "NumPy"
        # Test that operations use NumPy
        a = znp.array([1, 2, 3])
        assert isinstance(a, np.ndarray)

    # Verify backend is restored
    assert zb.get_backend().name == initial_backend

    # Test nested context managers
    with use_backend("numpy"):
        assert zb.get_backend().name == "NumPy"

        # Test JAX backend if available
        try:
            with use_backend("jax"):
                assert zb.get_backend().name == "JAX"

                # Go back to numpy
                with use_backend("numpy"):
                    assert zb.get_backend().name == "NumPy"

                # Back to JAX
                assert zb.get_backend().name == "JAX"

            # Back to numpy
            assert zb.get_backend().name == "NumPy"
        except ImportError:
            pass

    # Back to initial backend
    assert zb.get_backend().name == initial_backend

    # Test with invalid backend
    with pytest.raises(ValueError):
        with use_backend("invalid_backend"):
            pass


def test_host_callback_decorator():
    """Test the host_callback decorator."""

    # Define a host function
    def host_fn(x):
        return np.mean(x) * 2

    # Apply the decorator
    @host_callback(host_fn)
    def decorated_fn(x):
        return x

    # Test with regular array
    a = np.array([1, 2, 3, 4])
    result = decorated_fn(a)
    assert np.isclose(result, 5.0)  # 2.5 * 2 = 5.0

    # Test with backend array
    b = znp.array([1, 2, 3, 4])
    result = decorated_fn(b)
    assert np.isclose(result, 5.0)

    # Test with JAX array if available
    try:
        with use_backend("jax"):
            import warnings

            import jax.numpy as jnp

            from zfit2.backend.errors import NotImplementedInBackend

            # Ignore warnings about host_callback failures
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = jnp.array([1, 2, 3, 4])
                try:
                    result = decorated_fn(c)
                    assert np.isclose(result, 5.0)
                except (NotImplementedInBackend, ImportError) as e:
                    # Skip test if host_callback is not implemented in this JAX version
                    warnings.warn(f"Skipping JAX host_callback test: {e}")
    except ImportError:
        pass


def test_numpy_fallback_decorator():
    """Test the numpy_fallback decorator."""

    # Define a function that fails with SymPy but works with NumPy/JAX
    @numpy_fallback
    def complex_operation(x):
        # std operation might not be available in some backends
        return znp.std(x, ddof=1) * 2

    # Test with numpy backend - should work directly
    with use_backend("numpy"):
        a = znp.array([1, 2, 3, 4])
        result = complex_operation(a)
        assert np.isclose(result, 2.582, rtol=1e-3)  # ~1.291 * 2

    # Test with JAX backend if available - should also work
    try:
        with use_backend("jax"):
            a = znp.array([1, 2, 3, 4])
            result = complex_operation(a)
            assert np.isclose(result, 2.582, rtol=1e-3)
    except ImportError:
        pass

    # Test with SymPy backend if available - should fall back to NumPy
    try:
        with use_backend("sympy"):
            import sympy

            # This would fail with SymPy, but numpy_fallback should handle it
            a = znp.array([1, 2, 3, 4])
            result = complex_operation(a)
            assert np.isclose(result, 2.582, rtol=1e-3)
    except ImportError:
        pass


def test_nested_decorators():
    """Test using both decorators together."""

    def host_fn(x):
        return np.mean(x) * 2

    @host_callback(host_fn)
    @numpy_fallback
    def complex_decorated_fn(x):
        return znp.std(x, ddof=1)

    # The host_callback should apply to the result of numpy_fallback
    with use_backend("numpy"):
        a = znp.array([1, 2, 3, 4])
        result = complex_decorated_fn(a)
        assert np.isclose(result, 2.582, rtol=1e-3)  # 1.291 * 2

    # Try with JAX if available
    try:
        with use_backend("jax"):
            import warnings

            from zfit2.backend.errors import NotImplementedInBackend

            a = znp.array([1, 2, 3, 4])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = complex_decorated_fn(a)
                    assert np.isclose(result, 2.582, rtol=1e-3)
            except (NotImplementedInBackend, ImportError) as e:
                # Skip test if host_callback is not implemented in this JAX version
                warnings.warn(f"Skipping JAX nested decorators test: {e}")
    except ImportError:
        pass

    # Try with SymPy which would fail for std, but numpy_fallback handles it
    try:
        with use_backend("sympy"):
            a = znp.array([1, 2, 3, 4])
            result = complex_decorated_fn(a)
            assert np.isclose(result, 2.582, rtol=1e-3)
    except ImportError:
        pass
