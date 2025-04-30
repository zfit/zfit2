"""Comprehensive tests for the updated backend system with JAX tree utilities."""

from __future__ import annotations

import numpy as np
import pytest

from zfit2.backend import backend, numpy
from zfit2.backend.context import use_backend
from zfit2.backend.vectorize import auto_batch, vmap

# Skip tests if JAX is not available
try:
    import jax
    import jax.numpy as jnp
    from jax.tree import flatten, map, unflatten

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJAXBackend:
    """Test the updated JAX backend with the latest tree utilities."""

    def setup_method(self):
        """Set up the test by switching to JAX backend."""
        use_backend("jax").__enter__()

    def teardown_method(self):
        """Tear down the test by exiting the context."""
        use_backend("jax").__exit__(None, None, None)

    def test_array_creation(self):
        """Test array creation functions."""
        # Test array
        x = numpy.array([1.0, 2.0, 3.0])
        assert isinstance(x, jax.Array)
        assert np.array_equal(np.array(x), np.array([1.0, 2.0, 3.0]))

        # Test zeros
        z = numpy.zeros((2, 3))
        assert isinstance(z, jax.Array)
        assert z.shape == (2, 3)
        np.testing.assert_array_equal(np.array(z), 0.0)

        # Test ones
        o = numpy.ones((2, 3))
        assert isinstance(o, jax.Array)
        assert o.shape == (2, 3)
        np.testing.assert_array_equal(np.array(o), 1.0)

        # Test full
        f = numpy.full((2, 3), 5.0)
        assert isinstance(f, jax.Array)
        assert f.shape == (2, 3)
        np.testing.assert_array_equal(np.array(f), 5.0)

    def test_math_operations(self):
        """Test mathematical operations."""
        x = numpy.array([1.0, 2.0, 3.0])

        # Test exp
        exp_x = numpy.exp(x)
        np.testing.assert_allclose(exp_x, np.exp([1.0, 2.0, 3.0]))

        # Test log
        log_x = numpy.log(x)
        np.testing.assert_allclose(log_x, np.log([1.0, 2.0, 3.0]))

        # Test sin
        sin_x = numpy.sin(x)
        np.testing.assert_allclose(sin_x, np.sin([1.0, 2.0, 3.0]))

        # Test sum
        sum_x = numpy.sum(x)
        assert np.isclose(sum_x, 6.0)

    def test_tree_operations(self):
        """Test tree operations with latest JAX API."""
        # Create a nested structure (pytree)
        pytree = {
            "a": numpy.array([1.0, 2.0]),
            "b": (numpy.array([3.0]), numpy.array([4.0, 5.0])),
            "c": {"d": numpy.array([6.0, 7.0, 8.0])},
        }

        # Test tree_flatten
        leaves, treedef = backend.tree_flatten(pytree)
        assert len(leaves) == 4  # Should be 4 arrays

        # Test tree_map
        result = backend.tree_map(lambda x: x * 2, pytree)
        assert np.allclose(result["a"], np.array([2.0, 4.0]))
        assert np.allclose(result["b"][0], np.array([6.0]))
        assert np.allclose(result["b"][1], np.array([8.0, 10.0]))
        assert np.allclose(result["c"]["d"], np.array([12.0, 14.0, 16.0]))

        # Test tree_unflatten
        new_leaves = [
            numpy.array([10.0, 20.0]),  # a
            numpy.array([30.0]),  # b[0]
            numpy.array([40.0, 50.0]),  # b[1]
            numpy.array([60.0, 70.0, 80.0]),  # c[d]
        ]
        reconstructed = backend.tree_unflatten(treedef, new_leaves)
        assert np.allclose(reconstructed["a"], np.array([10.0, 20.0]))
        assert np.allclose(reconstructed["b"][0], np.array([30.0]))
        assert np.allclose(reconstructed["b"][1], np.array([40.0, 50.0]))
        assert np.allclose(reconstructed["c"]["d"], np.array([60.0, 70.0, 80.0]))

    def test_advanced_tree_operations(self):
        """Test more advanced tree operations."""
        # Create nested pytrees
        tree1 = {"a": numpy.array([1.0, 2.0]), "b": numpy.array([3.0, 4.0])}
        tree2 = {"a": numpy.array([5.0, 6.0]), "b": numpy.array([7.0, 8.0])}

        # Map function with multiple arguments
        result = backend.tree_map(lambda x, y: x + y, tree1, tree2)
        assert np.allclose(result["a"], np.array([6.0, 8.0]))
        assert np.allclose(result["b"], np.array([10.0, 12.0]))

    def test_differentiation(self):
        """Test differentiation operations."""

        def f(x):
            return numpy.sum(x**2)

        # Gradient
        grad_f = backend.grad(f)
        x = numpy.array([1.0, 2.0, 3.0])
        grad_result = grad_f(x)
        np.testing.assert_allclose(grad_result, np.array([2.0, 4.0, 6.0]))

        # Value and gradient
        val_and_grad_f = backend.value_and_grad(f)
        val, grad_val = val_and_grad_f(x)
        assert np.isclose(val, 14.0)
        np.testing.assert_allclose(grad_val, np.array([2.0, 4.0, 6.0]))

        # Hessian
        hess_f = backend.hessian(f)
        hess_result = hess_f(x)
        # Hessian should be diagonal with 2.0 on diagonal
        np.testing.assert_allclose(np.diag(hess_result), np.array([2.0, 2.0, 2.0]))

    def test_jit(self):
        """Test JIT compilation."""

        def f(x):
            return numpy.sum(x**2)

        # JIT compile the function
        jit_f = backend.jit(f)

        # Test the function
        x = numpy.array([1.0, 2.0, 3.0])
        result = jit_f(x)
        assert np.isclose(result, 14.0)

    def test_vectorization(self):
        """Test vectorization with vmap."""

        def f(x, y):
            return x * y

        # Use vmap directly
        vmap_f = backend.vmap(f)

        # Test the function
        x = numpy.array([1.0, 2.0, 3.0])
        y = numpy.array([4.0, 5.0, 6.0])
        result = vmap_f(x, y)
        np.testing.assert_allclose(result, np.array([4.0, 10.0, 18.0]))

        # Use the convenience wrapper
        vmap_f2 = vmap(f)
        result2 = vmap_f2(x, y)
        np.testing.assert_allclose(result2, np.array([4.0, 10.0, 18.0]))

        # Test auto_batch decorator
        @auto_batch(0)
        def add(x, y):
            return x + y

        # Apply to arrays
        batch_x = numpy.array([[1.0, 2.0], [3.0, 4.0]])
        batch_y = numpy.array([[10.0, 20.0], [30.0, 40.0]])
        batch_result = add(batch_x, batch_y)

        expected = numpy.array([[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_allclose(batch_result, expected)

    def test_control_flow(self):
        """Test control flow operations."""

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

        # Test scan
        def body_fun(carry, x):
            return carry + x, carry * x

        init = numpy.array(0.0)
        xs = numpy.array([1.0, 2.0, 3.0, 4.0])
        final, results = backend.scan(body_fun, init, xs)

        assert np.isclose(final, 10.0)  # 0 + 1 + 2 + 3 + 4 = 10
        np.testing.assert_allclose(
            results, np.array([0.0, 2.0, 9.0, 24.0])
        )  # 0*1, 1*2, 3*3, 6*4


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_backend_switching():
    """Test switching between backends."""
    # Create test data
    raw_data = [1.0, 2.0, 3.0]

    # Test with JAX backend
    with use_backend("jax"):
        x_jax = numpy.array(raw_data)
        assert isinstance(x_jax, jax.Array)

        # Use JAX-specific functionality
        jit_sum = backend.jit(lambda x: numpy.sum(x))
        sum_jax = jit_sum(x_jax)
        assert np.isclose(sum_jax, 6.0)

    # Test with NumPy backend
    with use_backend("numpy"):
        x_numpy = numpy.array(raw_data)
        assert isinstance(x_numpy, np.ndarray)

        # Sum the array
        sum_numpy = numpy.sum(x_numpy)
        assert np.isclose(sum_numpy, 6.0)

    # Back to JAX backend
    with use_backend("jax"):
        assert isinstance(numpy.array(raw_data), jax.Array)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_custom_pytree_node():
    """Test with custom pytree nodes."""
    # Only run if JAX is available
    if not HAS_JAX:
        return

    # Define a custom container
    class CustomContainer:
        def __init__(self, values):
            self.values = values

    # Register with JAX's tree registry
    try:
        from jax.tree import register_pytree_node

        # Define flatten and unflatten functions
        def container_flatten(container):
            return container.values, None

        def container_unflatten(_, values):
            return CustomContainer(values)

        # Register the custom container
        register_pytree_node(CustomContainer, container_flatten, container_unflatten)

        # Test with the custom container
        with use_backend("jax"):
            # Create a test pytree with the custom container
            custom = CustomContainer([numpy.array([1.0, 2.0]), numpy.array([3.0, 4.0])])

            # Test tree_map
            result = backend.tree_map(lambda x: x * 2, custom)
            assert isinstance(result, CustomContainer)
            assert len(result.values) == 2
            np.testing.assert_allclose(result.values[0], np.array([2.0, 4.0]))
            np.testing.assert_allclose(result.values[1], np.array([6.0, 8.0]))

            # Test tree_flatten and tree_unflatten
            leaves, treedef = backend.tree_flatten(custom)
            assert len(leaves) == 2

            new_leaves = [numpy.array([10.0, 20.0]), numpy.array([30.0, 40.0])]
            reconstructed = backend.tree_unflatten(treedef, new_leaves)
            assert isinstance(reconstructed, CustomContainer)
            assert len(reconstructed.values) == 2
            np.testing.assert_allclose(reconstructed.values[0], np.array([10.0, 20.0]))
            np.testing.assert_allclose(reconstructed.values[1], np.array([30.0, 40.0]))
    except ImportError:
        # Skip the test if jax.tree.register_pytree_node is not available
        pytest.skip("jax.tree.register_pytree_node not available")


if __name__ == "__main__":
    # Run tests if JAX is available
    if HAS_JAX:
        test_instance = TestJAXBackend()
        test_instance.setup_method()

        # Run test methods
        test_instance.test_array_creation()
        test_instance.test_math_operations()
        test_instance.test_tree_operations()
        test_instance.test_advanced_tree_operations()
        test_instance.test_differentiation()
        test_instance.test_jit()
        test_instance.test_vectorization()
        test_instance.test_control_flow()

        test_instance.teardown_method()

        # Run standalone tests
        test_backend_switching()
        test_custom_pytree_node()

        print("All tests passed!")
    else:
        print("JAX is not available. Skipping tests.")
