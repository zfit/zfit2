"""Tests for the backend system in zfit2."""

from __future__ import annotations

import numpy as np
import pytest

from zfit2.backend import (
    BackendError,
    NotImplementedInBackend,
    backend,
    get_backend,
    set_backend,
)
from zfit2.backend import numpy as znp


def test_backend_switching():
    """Test switching between backends."""
    # Test JAX backend
    try:
        set_backend("jax")
        assert get_backend().name == "JAX"
    except (ImportError, BackendError):
        pytest.skip("JAX not available")

    # Test NumPy backend
    set_backend("numpy")
    assert get_backend().name == "NumPy"

    # Test SymPy backend if available
    try:
        set_backend("sympy")
        assert get_backend().name == "SymPy"
    except (ImportError, BackendError):
        pass

    # Invalid backend
    with pytest.raises(ValueError):
        get_backend("invalid")


def test_array_creation():
    """Test array creation with different backends."""
    # Use NumPy backend for testing
    set_backend("numpy")

    # Create array
    a = znp.array([1, 2, 3])
    assert isinstance(a, np.ndarray)
    assert np.array_equal(a, np.array([1, 2, 3]))

    # Test JAX backend if available
    try:
        set_backend("jax")
        a = znp.array([1, 2, 3])
        assert np.array_equal(np.array(a), np.array([1, 2, 3]))
    except (ImportError, BackendError):
        pytest.skip("JAX not available")


def test_math_operations():
    """Test math operations with different backends."""
    # Use NumPy backend
    set_backend("numpy")

    # Test sqrt
    a = znp.array([1, 4, 9])
    b = znp.sqrt(a)
    np.testing.assert_allclose(b, np.array([1, 2, 3]))


def test_gradient():
    """Test gradient computation with different backends."""

    # Define a simple function
    def f(x):
        return x**2

    # Use NumPy backend
    set_backend("numpy")
    np_grad = backend.grad(f)
    assert np.isclose(np_grad(2.0), 4.0)

    # Use JAX backend if available
    try:
        set_backend("jax")
        jax_grad = backend.grad(f)
        assert np.isclose(jax_grad(2.0), 4.0)
    except (ImportError, BackendError):
        pytest.skip("JAX not available")


def test_hessian():
    """Test hessian computation with different backends."""

    # Define a simple function
    def f(x):
        return x**3

    # Use NumPy backend
    set_backend("numpy")
    np_hess = backend.hessian(f)
    assert np.isclose(np_hess(2.0), 12.0)

    # Use JAX backend if available
    try:
        set_backend("jax")
        jax_hess = backend.hessian(f)
        assert np.isclose(jax_hess(2.0), 12.0)
    except (ImportError, BackendError):
        pytest.skip("JAX not available")


def test_not_implemented():
    """Test handling of not implemented functions."""
    # Use SymPy backend
    try:
        set_backend("sympy")

        # Try a function that's not implemented
        with pytest.raises(NotImplementedInBackend):
            get_backend().clip([1, 2, 3], 1, 2)
    except (ImportError, BackendError):
        pytest.skip("SymPy not available")
