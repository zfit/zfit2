"""Tests for NLL offset functionality."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from zfit2.statistic import NLL, NLLOptions


class TestNLLOffset:
    """Test NLL with offset options."""

    def test_nll_no_offset(self):
        """Test NLL without offset (default behavior)."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        nll = NLL(
            dist, data, NLLOptions.none(), name="nll", label="Negative Log-Likelihood"
        )
        value = nll.value()

        # Should be regular NLL
        expected = -np.sum(dist.logpdf(data))
        assert value == pytest.approx(expected)

    def test_nll_mean_offset_default_start(self):
        """Test NLL with mean offset starting from 0."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean()
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        # First call triggers precompile
        value = nll.value()

        # Should start from 10000 (default for mean)
        assert value == pytest.approx(10000.0)

        # Verify offset was applied correctly
        logpdfs = dist.logpdf(data)
        mean_logpdf = np.mean(logpdfs)
        expected_without_adjustment = -np.sum(logpdfs - mean_logpdf)
        adjustment = 10000.0 - expected_without_adjustment
        assert float(nll._adjustment) == pytest.approx(adjustment, abs=1e-6)

    def test_nll_mean_offset_custom_start(self):
        """Test NLL with mean offset starting from custom value."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=10)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        # First call should give us 10
        value = nll.value()
        assert value == pytest.approx(10.0)

    def test_nll_median_offset(self):
        """Test NLL with median offset."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0, 2.0, -2.0])

        options = NLLOptions.median(start_value=5)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        value = nll.value()
        assert value == pytest.approx(5.0)

        # Verify median was used
        logpdfs = dist.logpdf(data)
        median_logpdf = np.median(logpdfs)
        assert np.allclose(nll._offset_values, median_logpdf)

    def test_nll_elementwise_offset(self):
        """Test NLL with elementwise offset."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.elementwise(start_value=100)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        value = nll.value()
        assert value == pytest.approx(100.0)

        # With elementwise offset, all logpdfs cancel out
        logpdfs = dist.logpdf(data)
        assert np.allclose(nll._offset_values, logpdfs)

    def test_nll_custom_offset_function(self):
        """Test NLL with custom offset function."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        def custom_offset(logpdfs):
            # Use max logpdf as offset
            return jnp.full_like(logpdfs, jnp.max(logpdfs))

        options = NLLOptions.custom(custom_offset)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        value = nll.value()

        # Verify custom function was used
        logpdfs = dist.logpdf(data)
        max_logpdf = np.max(logpdfs)
        expected = -np.sum(logpdfs - max_logpdf)
        assert value == pytest.approx(expected)

    def test_nll_offset_persistence(self):
        """Test that offset values persist across calls."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=20)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        # First call
        value1 = nll.value()
        assert value1 == pytest.approx(20.0)

        # Store offset values
        offset_values = nll._offset_values.copy()
        adjustment = nll._adjustment

        # Second call should use same offsets
        value2 = nll.value()
        assert value2 == pytest.approx(20.0)
        assert np.allclose(nll._offset_values, offset_values)
        assert nll._adjustment == adjustment

    def test_nll_offset_with_different_params(self):
        """Test NLL offset behavior when params change."""

        # Create a mock distribution that uses params
        class ParamDist:
            def __init__(self, base_loc):
                self.base_loc = base_loc

            def logpdf(self, x, params=None):
                loc = self.base_loc
                if params and "loc" in params:
                    loc = params["loc"]
                return scipy_stats.norm(loc=loc, scale=1).logpdf(x)

        dist = ParamDist(0.0)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=10)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        # First call with no params
        value1 = nll.value()
        assert value1 == pytest.approx(10.0)

        # Second call with different params - offset should not change
        value2 = nll.value(params={"loc": 0.5})
        assert value2 != pytest.approx(10.0)  # Different because logpdfs changed

        # But offset values should be the same (computed from initial call)
        assert nll._precompiled is True

    def test_nll_options_get_offset_method(self):
        """Test NLLOptions get_offset_method."""
        # Test with colon
        options = NLLOptions.mean(start_value=15.5)
        method = options.get_offset_method()
        assert method == "mean"
        assert options.start_value == 15.5

        # Test without colon
        options = NLLOptions.median()
        method = options.get_offset_method()
        assert method == "median"
        assert options.start_value == 10000.0

        # Test custom function
        def custom_fn(x):
            return x

        options = NLLOptions.custom(custom_fn)
        method = options.get_offset_method()
        assert method == "custom"

        # Test invalid value in string
        from zfit2.statistic.options import NLLOptionsLegacy

        with pytest.raises(ValueError, match="Invalid start value"):
            NLLOptionsLegacy(offset="mean:abc")

    def test_nll_with_multiple_distributions(self):
        """Test NLL offset with multiple distributions."""
        dist1 = scipy_stats.norm(0, 1)
        dist2 = scipy_stats.uniform(0, 2)

        data1 = np.array([0.0, 0.5])
        data2 = np.array([0.5, 1.0])

        options = NLLOptions.mean(start_value=50)
        nll = NLL(
            [dist1, dist2],
            [data1, data2],
            options=options,
            name="nll",
            label="Negative Log-Likelihood",
        )

        value = nll.value()
        assert value == pytest.approx(50.0)

    def test_nll_offset_with_jax_transformations(self):
        """Test NLL offset works with JAX transformations."""
        dist = scipy_stats.norm(0, 1)
        data = jnp.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=10)
        nll = NLL(
            dist, data, options=options, name="nll", label="Negative Log-Likelihood"
        )

        # Precompile
        nll.value()

        # Test with jit
        def compute_nll():
            return nll.value()

        jitted = jax.jit(compute_nll)
        result = jitted()
        assert result == pytest.approx(10.0)

    def test_nll_options_factory_methods(self):
        """Test NLLOptions factory methods."""
        # Test none
        opt_none = NLLOptions.none()
        assert opt_none.get_offset_method() == "none"
        assert opt_none.start_value == 0.0

        # Test mean
        opt_mean = NLLOptions.mean(start_value=10.0)
        assert opt_mean.get_offset_method() == "mean"
        assert opt_mean.start_value == 10.0

        # Test median
        opt_median = NLLOptions.median(start_value=20.0)
        assert opt_median.get_offset_method() == "median"
        assert opt_median.start_value == 20.0

        # Test elementwise
        opt_elem = NLLOptions.elementwise(start_value=100.0)
        assert opt_elem.get_offset_method() == "elementwise"
        assert opt_elem.start_value == 100.0

        # Test custom
        def custom_fn(x):
            return x * 2

        opt_custom = NLLOptions.custom(custom_fn)
        assert opt_custom.get_offset_method() == "custom"
        assert opt_custom.start_value == 0.0
