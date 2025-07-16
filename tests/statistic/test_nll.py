"""Tests for NLL statistic classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from zfit2.statistic import NLL, BaseNLL, NLLOptions


class TestBaseNLLStatistic:
    """Test BaseNLLStatistic abstract class."""

    def test_base_nll_abstract(self):
        """Test that BaseNLLStatistic is abstract."""
        with pytest.raises(TypeError):
            BaseNLL(name="test")

    def test_base_nll_implementation(self):
        """Test BaseNLLStatistic with concrete implementation."""

        class DummyNLL(BaseNLL):
            def _value(self, x=1.0, full=False):
                return x * 2.0

            def _loglike(self, params=None):
                return jnp.array([1.0, 2.0, 3.0])

            def _sum(self, ll):
                return jnp.sum(ll) * 2  # Custom sum

        nll = DummyNLL(name="dummy_nll")
        assert nll.name == "dummy_nll"

        # Test that value calls _value
        result = nll.value(x=5.0)
        assert result == 10.0

        # Test __call__
        result = nll()
        assert result == 2.0

        # Test loglike
        loglike = nll.loglike()
        np.testing.assert_allclose(loglike, np.array([1.0, 2.0, 3.0]))

        # Test sum
        summed = nll.sum(jnp.array([1.0, 2.0, 3.0]))
        assert summed == pytest.approx(12.0)  # (1+2+3)*2


class TestNLL:
    """Test NLL class."""

    def test_nll_accepts_collections(self):
        """Test that NLL accepts various collection types, not just lists."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Test with tuple
        nll_tuple = NLL(
            (dist,),
            (data,),
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll_tuple.dists) == 1
        assert len(nll_tuple.data) == 1

        # Test with custom collection class
        class CustomCollection:
            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

        custom_dists = CustomCollection([dist])
        custom_data = CustomCollection([data])
        nll_custom = NLL(
            custom_dists,
            custom_data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll_custom.dists) == 1
        assert len(nll_custom.data) == 1

        # Test with generator expression
        dists_gen = (d for d in [dist])
        data_gen = (d for d in [data])
        nll_gen = NLL(
            dists_gen,
            data_gen,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert len(nll_gen.dists) == 1
        assert len(nll_gen.data) == 1

    def test_nll_rejects_string_as_collection(self):
        """Test that strings are not treated as collections."""
        # String should be wrapped in a list, not treated as collection of chars
        nll = NLL(
            "dist",
            "data",
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert nll.dists == ["dist"]
        assert nll.data == ["data"]

    def test_nll_no_broadcasting_single_data_multiple_dists(self):
        """Test that single data with multiple dists raises error."""
        dists = [scipy_stats.norm(0, 1), scipy_stats.norm(1, 1)]
        data = np.array([1, 2, 3])  # Single dataset

        # Should raise error - no broadcasting
        with pytest.raises(
            ValueError,
            match="Number of distributions .* must exactly match .* No broadcasting is allowed",
        ):
            NLL(
                dists,
                data,
                options=NLLOptions.none(),
                name="nll",
                label="Negative Log-Likelihood",
            )

    def test_nll_no_broadcasting_single_dist_multiple_data(self):
        """Test that single dist with multiple data raises error."""
        dist = scipy_stats.norm(0, 1)
        data = [np.array([1, 2]), np.array([3, 4])]  # Multiple datasets

        # Should raise error - no broadcasting
        with pytest.raises(
            ValueError,
            match="Number of distributions .* must exactly match .* No broadcasting is allowed",
        ):
            NLL(
                dist,
                data,
                options=NLLOptions.none(),
                name="nll",
                label="Negative Log-Likelihood",
            )

    def test_nll_list_length_mismatch(self):
        """Test that mismatched lengths raise error (no broadcasting allowed)."""
        dists = [scipy_stats.norm(0, 1), scipy_stats.norm(1, 1)]
        data = [np.array([1, 2])]  # Only one data array for two dists

        with pytest.raises(
            ValueError,
            match="Number of distributions .* must exactly match .* No broadcasting is allowed",
        ):
            NLL(
                dists,
                data,
                options=NLLOptions.none(),
                name="nll",
                label="Negative Log-Likelihood",
            )

    def test_nll_multiple_dists_data_lists(self):
        """Test NLL with lists of distributions and corresponding data."""
        # Different distributions for different datasets
        dist1 = scipy_stats.norm(loc=0, scale=1)  # Standard normal
        dist2 = scipy_stats.uniform(loc=-1, scale=2)  # Uniform[-1, 1]

        # Different data for each distribution
        data1 = np.array([0.0, 0.5, -0.5])  # Data from normal
        data2 = np.array([-0.5, 0.0, 0.5])  # Data from uniform

        # Create NLL with lists
        nll = NLL(
            [dist1, dist2],
            [data1, data2],
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        # Verify storage
        assert len(nll.dists) == 2
        assert len(nll.data) == 2
        np.testing.assert_array_equal(nll.data[0], data1)
        np.testing.assert_array_equal(nll.data[1], data2)

        # Compute NLL
        result = nll.value()

        # Expected: -sum(log(p1(data1))) - sum(log(p2(data2)))
        expected = -np.sum(dist1.logpdf(data1)) - np.sum(dist2.logpdf(data2))
        assert result == pytest.approx(expected)

    def test_nll_creation(self):
        """Test NLL creation."""
        # Create scipy distributions and data
        dist = scipy_stats.norm(loc=0, scale=1)
        data = np.array([1, 2, 3])

        nll = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        assert nll.name == "nll"
        assert nll.label == "Negative Log-Likelihood"
        assert len(nll.dists) == 1
        assert nll.dists[0] == dist
        assert len(nll.data) == 1
        np.testing.assert_array_equal(nll.data[0], data)

    def test_nll_custom_name_label(self):
        """Test NLL with custom name and label."""
        nll = NLL(
            [], [], options=NLLOptions.none(), name="custom_nll", label="Custom NLL"
        )
        assert nll.name == "custom_nll"
        assert nll.label == "Custom NLL"

    def test_nll_loglike_no_dists(self):
        """Test loglike with no distributions."""
        nll = NLL(
            [],
            [],
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        loglike = nll.loglike()

        # Should return empty array
        assert isinstance(loglike, jnp.ndarray)
        assert loglike.shape == (0,) or loglike.size == 0

    def test_nll_loglike_scipy_dist(self):
        """Test loglike with scipy distribution."""
        # Create a normal distribution
        dist = scipy_stats.norm(loc=0, scale=1)
        data = np.array([0.0, 1.0, -1.0])

        nll = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        loglike = nll.loglike()

        # Compare with expected log pdf values
        expected = dist.logpdf(data)
        np.testing.assert_allclose(loglike, expected)

    def test_nll_loglike_multiple_dists(self):
        """Test loglike with multiple distributions and data."""
        # Create different distributions for different data
        dist1 = scipy_stats.norm(loc=0, scale=1)
        dist2 = scipy_stats.uniform(loc=0, scale=2)

        data1 = np.array([0.5, 1.0])
        data2 = np.array([0.5, 1.0])

        nll = NLL(
            [dist1, dist2],
            [data1, data2],
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )
        loglike = nll.loglike()

        # Should sum the log pdfs from both distributions
        expected = dist1.logpdf(data1) + dist2.logpdf(data2)
        np.testing.assert_allclose(loglike, expected)

    def test_nll_loglike_no_logpdf_method(self):
        """Test loglike with objects without logpdf method."""

        # Objects without logpdf method
        class BadDist:
            pass

        dist = BadDist()
        data = np.array([1, 2, 3])

        nll = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        # Should raise AttributeError
        with pytest.raises(AttributeError, match="does not have a logpdf method"):
            nll.loglike()

    def test_nll_sum(self):
        """Test sum method."""
        nll = NLL(
            [],
            [],
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        loglike_values = jnp.array([1.0, 2.0, 3.0])
        result = nll.sum(loglike_values)

        assert result == pytest.approx(6.0)

    def test_nll_value(self):
        """Test _value method (full computation)."""
        # Use scipy distribution
        dist = scipy_stats.norm(loc=0, scale=1)
        data = np.array([0.0, 1.0, 2.0])

        nll = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        # _value should return negative of sum of log pdfs
        expected_logpdfs = dist.logpdf(data)
        expected_value = -np.sum(expected_logpdfs)

        value = nll._value()
        assert value == pytest.approx(expected_value)

    def test_nll_call(self):
        """Test __call__ method."""
        dist = scipy_stats.expon(scale=1.0)
        data = np.array([0.5, 1.0])

        nll = NLL(
            dist,
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        # __call__ should work like value
        result = nll()
        expected = -np.sum(dist.logpdf(data))
        assert result == pytest.approx(expected)


class TestNLLPyTree:
    """Test JAX PyTree functionality for NLL."""

    def test_nll_pytree(self):
        """Test NLL PyTree registration."""
        dist = scipy_stats.norm(0, 1)
        data = jnp.array([1, 2, 3])

        nll = NLL(
            dist, data, options=NLLOptions.none(), name="test_nll", label="Test NLL"
        )

        # Flatten and unflatten
        flat, treedef = jax.tree_util.tree_flatten(nll)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        assert reconstructed.name == nll.name
        assert reconstructed.label == nll.label
        assert len(reconstructed.dists) == len(nll.dists)
        assert len(reconstructed.data) == len(nll.data)
        np.testing.assert_allclose(reconstructed.data[0], nll.data[0])

    def test_nll_jax_transformations(self):
        """Test NLL with JAX transformations."""

        class MockDist:
            def logpdf(self, data, params=None):
                # Use data in computation for gradient
                return -jnp.sum(data**2)

        # Use JAX array for data to enable gradients
        data = jnp.array([1.0, 2.0, 3.0])
        nll = NLL(
            [MockDist()],
            data,
            options=NLLOptions.none(),
            name="nll",
            label="Negative Log-Likelihood",
        )

        def loss_fn(data_array):
            # Create new NLL with modified data
            new_nll = NLL(
                nll.dists,
                data_array,
                options=NLLOptions.none(),
                name="nll",
                label="Negative Log-Likelihood",
            )
            return new_nll.value()

        # Test with jit
        jitted_loss = jax.jit(loss_fn)
        result = jitted_loss(data)
        assert isinstance(result, jnp.ndarray)

        # Test gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(data)
        assert grads.shape == data.shape
