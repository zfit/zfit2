"""Tests for UnbinnedData class."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zfit2.data.unbinned import UnbinnedData


class TestUnbinnedDataCreation:
    """Test UnbinnedData creation and basic properties."""

    def test_create_from_array(self):
        """Test creating UnbinnedData from array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        variables = ["x", "y"]

        unbinned = UnbinnedData(data, variables=variables)

        assert unbinned.shape == (3, 2)
        assert unbinned.n_samples == 3
        assert unbinned.n_variables == 2
        assert unbinned.variables == ["x", "y"]
        assert len(unbinned) == 3
        assert unbinned.weights is None

    def test_create_from_dict(self):
        """Test creating UnbinnedData from dictionary."""
        data_dict = {
            "x": np.array([1, 3, 5]),
            "y": np.array([2, 4, 6]),
        }

        unbinned = UnbinnedData(data_dict)

        assert unbinned.shape == (3, 2)
        assert unbinned.variables == ["x", "y"]
        np.testing.assert_allclose(unbinned["x"], jnp.array([1, 3, 5]))
        np.testing.assert_allclose(unbinned["y"], jnp.array([2, 4, 6]))

    def test_create_with_weights(self):
        """Test creating UnbinnedData with weights."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.array([0.5, 1.0, 1.5])

        unbinned = UnbinnedData(data, variables=["x", "y"], weights=weights)

        assert unbinned.weights is not None
        np.testing.assert_allclose(unbinned.weights, weights)

    def test_create_1d_array(self):
        """Test creating UnbinnedData from 1D array."""
        data = np.array([1, 2, 3, 4])

        unbinned = UnbinnedData(data)

        assert unbinned.shape == (4, 1)
        assert unbinned.variables == ["var_0"]

    def test_default_variable_names(self):
        """Test default variable names generation."""
        data = np.random.randn(10, 3)

        unbinned = UnbinnedData(data)

        assert unbinned.variables == ["var_0", "var_1", "var_2"]

    def test_invalid_shapes(self):
        """Test error handling for invalid shapes."""
        # Wrong weight shape
        with pytest.raises(ValueError, match="weights shape"):
            UnbinnedData(np.array([[1, 2]]), weights=np.array([1, 2]))

        # Wrong number of variables
        with pytest.raises(ValueError, match="Number of variables"):
            UnbinnedData(np.array([[1, 2]]), variables=["x", "y", "z"])

        # Dict with variables argument
        with pytest.raises(ValueError, match="variables argument is ignored"):
            UnbinnedData({"x": [1, 2]}, variables=["x"])

        # 3D array
        with pytest.raises(ValueError, match="must be 1D or 2D array"):
            UnbinnedData(np.ones((2, 3, 4)))


class TestUnbinnedDataAccess:
    """Test data access methods."""

    def test_getitem_by_name(self):
        """Test accessing data by variable name."""
        data = UnbinnedData({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})

        x_data = data["x"]
        np.testing.assert_allclose(x_data, jnp.array([1, 2, 3]))

        # Non-existent variable
        with pytest.raises(KeyError, match="Variable 'w' not found"):
            data["w"]

    def test_getitem_by_index(self):
        """Test accessing data by index."""
        data = UnbinnedData([[1, 2, 3], [4, 5, 6]], variables=["a", "b", "c"])

        np.testing.assert_allclose(data[0], jnp.array([1, 4]))
        np.testing.assert_allclose(data[1], jnp.array([2, 5]))
        np.testing.assert_allclose(data[2], jnp.array([3, 6]))

    def test_getitem_by_slice(self):
        """Test accessing data by slice."""
        data = UnbinnedData([[1, 2, 3], [4, 5, 6]], variables=["a", "b", "c"])

        sliced = data[1:3]
        assert isinstance(sliced, UnbinnedData)
        assert sliced.shape == (2, 2)
        assert sliced.variables == ["b", "c"]

    def test_getitem_by_list(self):
        """Test accessing data by list of names."""
        data = UnbinnedData({"x": [1, 2], "y": [3, 4], "z": [5, 6]})

        subset = data[["x", "z"]]
        assert isinstance(subset, UnbinnedData)
        assert subset.shape == (2, 2)
        assert subset.variables == ["x", "z"]
        np.testing.assert_allclose(subset.data, jnp.array([[1, 5], [2, 6]]))

        # With non-existent variable
        with pytest.raises(KeyError, match="Variable 'w' not found"):
            data[["x", "w"]]

    def test_getitem_invalid_type(self):
        """Test error for invalid key type."""
        data = UnbinnedData([[1, 2]])

        with pytest.raises(TypeError, match="Invalid key type"):
            data[None]

    def test_to_dict(self):
        """Test converting to dictionary."""
        data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])

        data_dict = data.to_dict()
        assert set(data_dict.keys()) == {"x", "y"}
        np.testing.assert_allclose(data_dict["x"], jnp.array([1, 3]))
        np.testing.assert_allclose(data_dict["y"], jnp.array([2, 4]))


class TestUnbinnedDataString:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__ method."""
        data = UnbinnedData([[1, 2]], variables=["x", "y"])
        repr_str = repr(data)
        assert "UnbinnedData" in repr_str
        assert "shape=(1, 2)" in repr_str
        assert "variables=['x', 'y']" in repr_str

        # With weights
        data_weighted = UnbinnedData([[1, 2]], variables=["x", "y"], weights=[0.5])
        repr_str = repr(data_weighted)
        assert "weights=(1,)" in repr_str

    def test_str(self):
        """Test __str__ method."""
        data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])
        str_repr = str(data)

        assert "UnbinnedData with 2 samples and 2 variables" in str_repr
        assert "Variables: x, y" in str_repr
        assert "First few samples:" in str_repr

        # With weights
        data_weighted = UnbinnedData([[1, 2]], variables=["x", "y"], weights=[0.5])
        str_repr = str(data_weighted)
        assert "Weights: yes" in str_repr

    def test_str_many_samples(self):
        """Test string representation with many samples."""
        data = UnbinnedData(np.random.randn(100, 3))
        str_repr = str(data)

        assert "... (95 more rows)" in str_repr


class TestUnbinnedDataArithmetic:
    """Test arithmetic operations."""

    def test_add_scalar(self):
        """Test adding a scalar."""
        data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])
        result = data + 10

        assert isinstance(result, UnbinnedData)
        np.testing.assert_allclose(result.data, jnp.array([[11, 12], [13, 14]]))
        assert result.variables == data.variables

    def test_add_array(self):
        """Test adding an array."""
        data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])
        result = data + jnp.array([10, 20])

        np.testing.assert_allclose(result.data, jnp.array([[11, 22], [13, 24]]))

    def test_add_unbinned(self):
        """Test adding two UnbinnedData objects."""
        data1 = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])
        data2 = UnbinnedData([[10, 20], [30, 40]], variables=["x", "y"])

        result = data1 + data2
        np.testing.assert_allclose(result.data, jnp.array([[11, 22], [33, 44]]))

        # Different shapes
        data3 = UnbinnedData([[1, 2]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            data1 + data3

        # Different variables
        data4 = UnbinnedData([[1, 2], [3, 4]], variables=["a", "b"])
        with pytest.raises(ValueError, match="Variable mismatch"):
            data1 + data4

    def test_subtract(self):
        """Test subtraction."""
        data = UnbinnedData([[10, 20]], variables=["x", "y"])

        # Scalar
        result = data - 5
        np.testing.assert_allclose(result.data, jnp.array([[5, 15]]))

        # UnbinnedData
        data2 = UnbinnedData([[1, 2]], variables=["x", "y"])
        result = data - data2
        np.testing.assert_allclose(result.data, jnp.array([[9, 18]]))

    def test_multiply(self):
        """Test multiplication."""
        data = UnbinnedData([[2, 3]], variables=["x", "y"])

        # Scalar
        result = data * 5
        np.testing.assert_allclose(result.data, jnp.array([[10, 15]]))

        # UnbinnedData
        data2 = UnbinnedData([[2, 3]], variables=["x", "y"])
        result = data * data2
        np.testing.assert_allclose(result.data, jnp.array([[4, 9]]))

    def test_divide(self):
        """Test division."""
        data = UnbinnedData([[10, 20]], variables=["x", "y"])

        # Scalar
        result = data / 2
        np.testing.assert_allclose(result.data, jnp.array([[5, 10]]))

        # UnbinnedData
        data2 = UnbinnedData([[2, 4]], variables=["x", "y"])
        result = data / data2
        np.testing.assert_allclose(result.data, jnp.array([[5, 5]]))

    def test_power(self):
        """Test power operation."""
        data = UnbinnedData([[2, 3]], variables=["x", "y"])
        result = data**2

        np.testing.assert_allclose(result.data, jnp.array([[4, 9]]))

    def test_reverse_operations(self):
        """Test reverse arithmetic operations."""
        data = UnbinnedData([[2, 4]], variables=["x", "y"])

        # Reverse add
        result = 10 + data
        np.testing.assert_allclose(result.data, jnp.array([[12, 14]]))

        # Reverse subtract
        result = 10 - data
        np.testing.assert_allclose(result.data, jnp.array([[8, 6]]))

        # Reverse multiply
        result = 3 * data
        np.testing.assert_allclose(result.data, jnp.array([[6, 12]]))

        # Reverse divide
        result = 12 / data
        np.testing.assert_allclose(result.data, jnp.array([[6, 3]]))

    def test_weights_preserved(self):
        """Test that weights are preserved in operations."""
        data = UnbinnedData([[1, 2]], variables=["x", "y"], weights=[0.5])

        result = data + 10
        assert result.weights is not None
        np.testing.assert_allclose(result.weights, jnp.array([0.5]))


@pytest.mark.parametrize(
    ("op", "scalar", "expected"),
    [
        (lambda d, s: d + s, 10, [[11, 12], [13, 14]]),
        (lambda d, s: d - s, 5, [[-4, -3], [-2, -1]]),
        (lambda d, s: d * s, 2, [[2, 4], [6, 8]]),
        (lambda d, s: d / s, 2, [[0.5, 1], [1.5, 2]]),
        (lambda d, s: d**s, 2, [[1, 4], [9, 16]]),
    ],
)
def test_unbinned_arithmetic_operations(op, scalar, expected):
    """Test various arithmetic operations with parametrize."""
    data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"])
    result = op(data, scalar)

    assert isinstance(result, UnbinnedData)
    np.testing.assert_allclose(result.data, jnp.array(expected))
    assert result.variables == data.variables


class TestUnbinnedDataPyTree:
    """Test JAX PyTree functionality."""

    def test_pytree_flatten_unflatten(self):
        """Test PyTree flattening and unflattening."""
        data = UnbinnedData([[1, 2], [3, 4]], variables=["x", "y"], weights=[0.5, 1.0])

        # Flatten and unflatten
        flat, treedef = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        # Check reconstruction
        np.testing.assert_allclose(reconstructed.data, data.data)
        assert reconstructed.variables == data.variables
        np.testing.assert_allclose(reconstructed.weights, data.weights)

    def test_pytree_without_weights(self):
        """Test PyTree without weights."""
        data = UnbinnedData([[1, 2]], variables=["x", "y"])

        flat, treedef = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        np.testing.assert_allclose(reconstructed.data, data.data)
        assert reconstructed.variables == data.variables
        assert reconstructed.weights is None

    def test_jax_transformations(self):
        """Test with JAX transformations."""

        def sum_data(unbinned):
            return jnp.sum(unbinned.data)

        data = UnbinnedData([[1.0, 2.0], [3.0, 4.0]], variables=["x", "y"])

        # Test with jit
        jitted_sum = jax.jit(sum_data)
        result = jitted_sum(data)
        assert result == pytest.approx(10.0)

        # Test with grad
        def loss_fn(unbinned):
            return jnp.sum(unbinned.data**2)

        grad_fn = jax.grad(loss_fn)
        grad_data = grad_fn(data)

        assert isinstance(grad_data, UnbinnedData)
        np.testing.assert_allclose(grad_data.data, 2 * data.data)
