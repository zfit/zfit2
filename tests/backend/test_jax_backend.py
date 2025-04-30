"""Tests for the JAX backend with updated APIs."""

from __future__ import annotations

import numpy as np
import pytest

import zfit2.backend as zb
from zfit2.backend import backend, numpy
from zfit2.backend.context import use_backend

# Skip tests if JAX is not available
try:
    import jax
    import jax.numpy as jnp

    # Use jax.tree instead of jax.tree_util
    from jax.tree import flatten as tree_flatten
    from jax.tree import map as tree_map
    from jax.tree import unflatten as tree_unflatten

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_backend_imports():
    """Test that the JAX backend can be imported and used."""
    with use_backend("jax"):
        # Make sure we're using the JAX backend
        assert zb.get_backend().name == "JAX"

        # Test basic array creation
        x = numpy.array([1.0, 2.0, 3.0])
        assert isinstance(x, jax.Array)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_tree_utils():
    """Test the JAX tree utilities in the backend."""
    with use_backend("jax"):
        # Create a test pytree
        pytree = {
            "a": numpy.array([1.0, 2.0]),
            "b": (numpy.array([3.0]), numpy.array([4.0, 5.0])),
        }

        # Test tree_flatten
        leaves, treedef = backend.tree_flatten(pytree)
        assert len(leaves) == 3  # Should be 3 arrays

        # Test tree_map
        result = backend.tree_map(lambda x: x * 2, pytree)
        assert np.allclose(result["a"], np.array([2.0, 4.0]))
        assert np.allclose(result["b"][0], np.array([6.0]))

        # Test tree_unflatten
        new_leaves = [
            numpy.array([10.0, 20.0]),
            numpy.array([30.0]),
            numpy.array([40.0, 50.0]),
        ]
        reconstructed = backend.tree_unflatten(treedef, new_leaves)
        assert np.allclose(reconstructed["a"], np.array([10.0, 20.0]))
        assert np.allclose(reconstructed["b"][0], np.array([30.0]))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_differentiation():
    """Test differentiation with the JAX backend."""
    with use_backend("jax"):

        def f(x):
            return numpy.sum(x**2)

        # Test grad
        grad_f = backend.grad(f)
        x = numpy.array([1.0, 2.0, 3.0])
        grad_result = grad_f(x)
        assert np.allclose(grad_result, np.array([2.0, 4.0, 6.0]))

        # Test value_and_grad
        val_and_grad_f = backend.value_and_grad(f)
        val, grad_val = val_and_grad_f(x)
        assert np.isclose(val, 14.0)
        assert np.allclose(grad_val, np.array([2.0, 4.0, 6.0]))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_jit():
    """Test JIT compilation with the JAX backend."""
    with use_backend("jax"):

        def f(x):
            return numpy.sum(x**2)

        # JIT compile the function
        jit_f = backend.jit(f)

        # Test the function
        x = numpy.array([1.0, 2.0, 3.0])
        result = jit_f(x)
        assert np.isclose(result, 14.0)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_vmap():
    """Test vectorization with the JAX backend."""
    with use_backend("jax"):

        def f(x, y):
            return x * y

        # Use vmap to vectorize the function
        vmap_f = backend.vmap(f)

        # Test the function
        x = numpy.array([1.0, 2.0, 3.0])
        y = numpy.array([4.0, 5.0, 6.0])
        result = vmap_f(x, y)
        assert np.allclose(result, np.array([4.0, 10.0, 18.0]))


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_scan():
    """Test the scan operation with the JAX backend."""
    with use_backend("jax"):

        def body_fun(carry, x):
            return carry + x, carry * x

        # Use scan to apply the function
        init = numpy.array(0.0)
        xs = numpy.array([1.0, 2.0, 3.0, 4.0])
        final, results = backend.scan(body_fun, init, xs)

        assert np.isclose(final, 10.0)  # 0 + 1 + 2 + 3 + 4 = 10
        assert np.allclose(
            results, np.array([0.0, 2.0, 9.0, 24.0])
        )  # 0*1, 1*2, 3*3, 6*4


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_control_flow():
    """Test control flow operations with the JAX backend."""
    with use_backend("jax"):
        # Test cond
        def f_cond(x):
            return backend.cond(x > 0, lambda x: x**2, lambda x: -x, x)

        assert np.isclose(f_cond(2.0), 4.0)
        assert np.isclose(f_cond(-2.0), 2.0)

        # Test while_loop
        def f_while(x):
            def cond_fun(val):
                return val < 10

            def body_fun(val):
                return val * 2

            return backend.while_loop(cond_fun, body_fun, x)

        assert np.isclose(f_while(1.0), 16.0)  # 1 -> 2 -> 4 -> 8 -> 16 (stop)

        # Test fori_loop
        def f_fori(x):
            def body_fun(i, val):
                return val + i

            return backend.fori_loop(0, 5, body_fun, x)

        assert np.isclose(f_fori(0.0), 10.0)  # 0 + 0 + 1 + 2 + 3 + 4 = 10
