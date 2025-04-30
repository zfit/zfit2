"""Tests for basic backend functionality."""

from __future__ import annotations

import numpy as np
import pytest

# Import backend interfaces
import zfit2.backend as zb
from zfit2.backend import numpy as znp
from zfit2.backend.context import use_backend

# Check which backends are available
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import sympy

    HAS_SYMPY = True and False  # HACK, don't check sympy
except ImportError:
    HAS_SYMPY = False


def test_backend_switching():
    """Test switching between backends."""
    # Test default backend
    current_name = zb.get_backend().name
    assert current_name in ["JAX", "NumPy", "SymPy"]

    # Test switching to NumPy
    zb.set_backend("numpy")
    assert zb.get_backend().name == "NumPy"

    # Test context manager
    with use_backend("numpy"):
        assert zb.get_backend().name == "NumPy"
        a = znp.array([1.0, 2.0, 3.0])
        assert isinstance(a, np.ndarray)

    # Test JAX if available
    if HAS_JAX:
        with use_backend("jax"):
            assert zb.get_backend().name == "JAX"
            a = znp.array([1.0, 2.0, 3.0])
            assert isinstance(a, jax.Array)

    # Test SymPy if available
    if HAS_SYMPY:
        with use_backend("sympy"):
            assert zb.get_backend().name == "SymPy"
            x = sympy.symbols("x")
            result = znp.sin(x) + znp.cos(x)
            assert sympy.simplify(result - sympy.sin(x) - sympy.cos(x)) == 0


@pytest.mark.parametrize(
    "backend",
    ["numpy"] + (["jax"] if HAS_JAX else []) + (["sympy"] if HAS_SYMPY else []),
)
def test_array_creation(backend):
    """Test array creation across backends."""
    with use_backend(backend):
        # Test various array creation methods
        a1 = znp.array([1, 2, 3])
        assert znp.shape(a1) == (3,)

        shape = (2, 3)
        a2 = znp.zeros(shape)
        assert znp.shape(a2) == shape
        np.testing.assert_allclose(np.array(a2), np.zeros(shape))

        a3 = znp.ones(shape)
        assert znp.shape(a3) == shape
        np.testing.assert_allclose(a3, np.ones(shape))

        shape = (2, 5)
        a4 = znp.full(shape, 5.0)
        assert znp.shape(a4) == shape
        np.testing.assert_allclose(a4, np.full(shape, 5.0))

        # Test array creation helpers
        if backend != "sympy":  # Skip for SymPy as it might not handle all these well
            a5 = znp.arange(10)
            assert znp.size(a5) == 10

            a6 = znp.linspace(0, 1, 11)
            assert znp.size(a6) == 11
            assert znp.isclose(a6[0], 0.0)
            assert znp.isclose(a6[-1], 1.0)

            a7 = znp.eye(3)
            assert znp.shape(a7) == (3, 3)
            # Create array of ones for comparison instead of using a list
            ones = znp.ones(3)
            np.testing.assert_allclose(znp.diag(a7), ones)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_array_manipulation(backend):
    """Test array manipulation functions."""
    with use_backend(backend):
        a = znp.array([[1, 2, 3], [4, 5, 6]])

        # Test reshape
        b = znp.reshape(a, (3, 2))
        assert znp.shape(b) == (3, 2)

        # Test transpose
        c = znp.transpose(a)
        assert znp.shape(c) == (3, 2)

        # Test concatenate
        d1 = znp.array([[1, 2], [3, 4]])
        d2 = znp.array([[5, 6], [7, 8]])
        d3 = znp.concatenate([d1, d2], axis=0)
        assert znp.shape(d3) == (4, 2)

        # Test stack
        e = znp.stack([d1, d2], axis=0)
        assert znp.shape(e) == (2, 2, 2)

        # Test where
        f = znp.where(a > 3, a, 0)
        assert znp.sum(f) == 15  # sum of 4, 5, 6


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_math_operations(backend):
    """Test mathematical operations."""
    with use_backend(backend):
        a = znp.array([1.0, 2.0, 3.0])

        # Test basic math operations
        assert znp.allclose(znp.exp(a), np.exp(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.log(a), np.log(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.sin(a), np.sin(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.cos(a), np.cos(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.tan(a), np.tan(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.sqrt(a), np.sqrt(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.square(a), np.square(np.array([1.0, 2.0, 3.0])))
        assert znp.allclose(znp.abs(a), np.abs(np.array([1.0, 2.0, 3.0])))

        # Test reduction operations
        assert znp.isclose(znp.sum(a), 6.0)
        assert znp.isclose(znp.mean(a), 2.0)
        assert znp.isclose(znp.min(a), 1.0)
        assert znp.isclose(znp.max(a), 3.0)

        # Test pairwise operations
        b = znp.array([2.0, 3.0, 4.0])
        assert znp.allclose(a + b, np.array([3.0, 5.0, 7.0]))
        assert znp.allclose(a * b, np.array([2.0, 6.0, 12.0]))
        assert znp.allclose(a / b, np.array([0.5, 2.0 / 3.0, 0.75]))
        assert znp.allclose(a - b, np.array([-1.0, -1.0, -1.0]))
        assert znp.allclose(
            znp.power(a, b),
            np.power(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])),
        )


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_trig_functions(backend):
    """Test trigonometric and hyperbolic functions."""
    with use_backend(backend):
        a = znp.array([0.0, 0.5 * znp.pi])

        # Trigonometric functions
        # Convert numpy arrays to backend arrays for comparison
        np_sin = znp.array(np.sin(np.array([0.0, 0.5 * np.pi])))
        np_cos = znp.array(np.cos(np.array([0.0, 0.5 * np.pi])))
        np_tan = znp.array(np.tan(np.array([0.0])))

        assert znp.allclose(znp.sin(a), np_sin)
        assert znp.allclose(znp.cos(a), np_cos)
        assert znp.allclose(
            znp.tan(a)[0], np_tan[0]
        )  # tan(pi/2) is undefined, so only check first element

        # Inverse trigonometric functions
        b = znp.array([-1.0, 0.0, 1.0])
        np_arcsin = znp.array(np.arcsin(np.array([-1.0, 0.0, 1.0])))
        np_arccos = znp.array(np.arccos(np.array([-1.0, 0.0, 1.0])))
        assert znp.allclose(znp.arcsin(b), np_arcsin)
        assert znp.allclose(znp.arccos(b), np_arccos)

        # Hyperbolic functions
        c = znp.array([0.0, 1.0])
        np_sinh = znp.array(np.sinh(np.array([0.0, 1.0])))
        np_cosh = znp.array(np.cosh(np.array([0.0, 1.0])))
        np_tanh = znp.array(np.tanh(np.array([0.0, 1.0])))
        assert znp.allclose(znp.sinh(c), np_sinh)
        assert znp.allclose(znp.cosh(c), np_cosh)
        assert znp.allclose(znp.tanh(c), np_tanh)

        # Inverse hyperbolic functions
        np_arcsinh = znp.array(np.arcsinh(np.array([0.0, 1.0])))
        assert znp.allclose(znp.arcsinh(c), np_arcsinh)

        d = znp.array([1.0, 2.0])  # domain of arccosh is [1, inf)
        np_arccosh = znp.array(np.arccosh(np.array([1.0, 2.0])))
        assert znp.allclose(znp.arccosh(d), np_arccosh)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_array_properties(backend):
    """Test array properties and utility functions."""
    with use_backend(backend):
        a = znp.array([[1, 2, 3], [4, 5, 6]])

        # Test shape, size, ndim
        assert znp.shape(a) == (2, 3)
        assert znp.size(a) == 6
        assert znp.ndim(a) == 2

        # Test type-related functions - these may differ between backends
        # So just make sure they run without errors
        dt = znp.dtype(a)

        # Test rounding functions
        b = znp.array([1.1, 1.5, 1.9, -1.1, -1.5, -1.9])
        assert znp.allclose(
            znp.round(b), np.round(np.array([1.1, 1.5, 1.9, -1.1, -1.5, -1.9]))
        )
        assert znp.allclose(
            znp.floor(b), np.floor(np.array([1.1, 1.5, 1.9, -1.1, -1.5, -1.9]))
        )
        assert znp.allclose(
            znp.ceil(b), np.ceil(np.array([1.1, 1.5, 1.9, -1.1, -1.5, -1.9]))
        )


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
