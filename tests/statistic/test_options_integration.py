"""Integration tests for NLLOptions with NLL."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from zfit2.statistic import NLL
from zfit2.statistic.options import NLLOptions


class TestNLLOptionsWithNLL:
    """Test NLLOptions integration with NLL class."""

    def test_nll_with_statistic_options(self):
        """Test NLL with new NLLOptions."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Test with default options
        options = NLLOptions.default()
        nll = NLL(dist, data, options=options)
        value = nll()
        assert isinstance(value, float | jnp.ndarray)

        # Test with chained options
        options2 = NLLOptions().offset("median", start_value=5000).sum("standard")
        nll2 = NLL(dist, data, options=options2)
        value2 = nll2()
        assert isinstance(value2, float | jnp.ndarray)

        # Different offset methods should give different results
        assert value != value2

    def test_nll_with_custom_offset_function(self):
        """Test NLL with custom offset function using NLLOptions."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Custom offset function
        def percentile_90(values):
            return jnp.percentile(values, 90)

        options = NLLOptions().offset(percentile_90)
        nll = NLL(dist, data, options=options)
        value = nll()

        # Should work without errors
        assert isinstance(value, float | jnp.ndarray)

    def test_nll_precompilation_with_options(self):
        """Test that NLL precompilation works with various options."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Test different offset methods
        for method in ["none", "mean", "median", "elementwise"]:
            options = NLLOptions().offset(method, start_value=1000)
            nll = NLL(dist, data, options=options)

            # First call should trigger precompilation
            assert not nll._precompiled
            value1 = nll()
            assert nll._precompiled

            # Second call should use precompiled values
            value2 = nll()
            assert jnp.allclose(value1, value2)

    def test_statistic_options_immutability_in_nll(self):
        """Test that options remain immutable when used in NLL."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([1, 2, 3])

        # Create base options
        base_options = NLLOptions()

        # Use in one NLL
        options1 = base_options.offset("mean", start_value=1000)
        nll1 = NLL(dist, data, options=options1)

        # Use in another NLL with different config
        options2 = base_options.offset("median", start_value=2000)
        nll2 = NLL(dist, data, options=options2)

        # Both should work independently
        value1 = nll1()
        value2 = nll2()

        # Values should be different due to different offsets
        assert value1 != value2

        # Original options should be unchanged
        assert base_options.get_offset_config() is None

    def test_complex_chaining_example(self):
        """Test complex chaining with multiple distributions."""
        # Multiple distributions and data
        dists = [scipy_stats.norm(0, 1), scipy_stats.norm(2, 1.5)]
        data = [np.array([0.5, 1.0, 1.5]), np.array([1.5, 2.0, 2.5])]

        # Complex chained configuration
        options = (
            NLLOptions().offset("mean", start_value=20000).sum("standard")
        )  # Ready for future sum methods

        nll = NLL(dists, data, options=options)
        value = nll()

        # Should start around 20000
        assert 19000 < value < 21000  # Some tolerance for numerical differences

    def test_options_validation_in_nll(self):
        """Test that invalid options are caught early."""
        scipy_stats.norm(0, 1)
        np.array([1, 2, 3])

        # Invalid offset method should raise error during options creation
        with pytest.raises(ValueError, match="Invalid offset method"):
            NLLOptions().offset("invalid_method")

        # Invalid sum method
        with pytest.raises(ValueError, match="Invalid sum method"):
            NLLOptions().sum("invalid_sum")
