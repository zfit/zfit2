"""Tests for NLL full parameter functionality."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as scipy_stats

from zfit2.statistic import NLL
from zfit2.statistic.options import NLLOptions


class TestNLLFullParameter:
    """Test NLL full parameter functionality."""

    def test_full_parameter_no_offset(self):
        """Test that full=True gives raw NLL without offset."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        # Create NLL with mean offset
        options = NLLOptions.mean(start_value=10000)
        nll = NLL(dist, data, options=options)

        # Regular value should be offset
        value_offset = nll.value()
        assert value_offset == pytest.approx(10000.0)

        # Full value should be raw NLL
        value_full = nll.value(full=True)
        expected_raw = -np.sum(dist.logpdf(data))
        assert value_full == pytest.approx(expected_raw)

        # Check they're different
        assert value_offset != value_full

    def test_full_parameter_with_different_offsets(self):
        """Test full parameter with various offset methods."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0, 2.0])

        # Expected raw NLL
        expected_raw = -np.sum(dist.logpdf(data))

        # Test with different offset methods
        for method, start_value in [
            ("none", 0),
            ("mean", 5000),
            ("median", 3000),
            ("elementwise", 1000),
        ]:
            options = getattr(NLLOptions, method)(start_value=start_value)
            nll = NLL(dist, data, options=options)

            # Full value should always be the same (raw NLL)
            value_full = nll.value(full=True)
            assert value_full == pytest.approx(expected_raw)

            # Offset value should be different (except for "none" with start_value=0)
            value_offset = nll.value()
            if method == "none" and start_value == 0:
                assert value_offset == pytest.approx(expected_raw)
            else:
                assert value_offset != pytest.approx(expected_raw)

    def test_full_parameter_no_precompilation(self):
        """Test that full=True doesn't trigger precompilation."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=1000)
        nll = NLL(dist, data, options=options)

        # Call with full=True first
        assert not nll._precompiled
        nll.value(full=True)

        # Should not have precompiled
        assert not nll._precompiled

        # Now call regular value
        value_offset = nll.value()

        # Now it should be precompiled
        assert nll._precompiled
        assert value_offset == pytest.approx(1000.0)

    def test_full_parameter_multiple_distributions(self):
        """Test full parameter with multiple distributions."""
        dist1 = scipy_stats.norm(0, 1)
        dist2 = scipy_stats.uniform(0, 2)

        data1 = np.array([0.0, 0.5])
        data2 = np.array([0.5, 1.0])

        options = NLLOptions.median(start_value=2000)
        nll = NLL([dist1, dist2], [data1, data2], options=options)

        # Full value
        value_full = nll.value(full=True)
        expected_raw = -(np.sum(dist1.logpdf(data1)) + np.sum(dist2.logpdf(data2)))
        assert value_full == pytest.approx(expected_raw)

        # Offset value
        value_offset = nll.value()
        assert value_offset == pytest.approx(2000.0)

    def test_full_parameter_with_custom_offset(self):
        """Test full parameter with custom offset function."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        def custom_offset(logpdfs):
            # Use 75th percentile
            return jnp.full_like(logpdfs, jnp.percentile(logpdfs, 75))

        options = NLLOptions.custom(custom_offset)
        nll = NLL(dist, data, options=options)

        # Full value should be raw
        value_full = nll.value(full=True)
        expected_raw = -np.sum(dist.logpdf(data))
        assert value_full == pytest.approx(expected_raw)

        # Offset value should be different
        value_offset = nll.value()
        assert value_offset != pytest.approx(expected_raw)

    def test_full_parameter_consistency(self):
        """Test that multiple calls with full=True give consistent results."""
        dist = scipy_stats.norm(0, 1)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=5000)
        nll = NLL(dist, data, options=options)

        # Multiple calls with full=True
        values_full = [nll.value(full=True) for _ in range(5)]

        # All should be identical
        for v in values_full[1:]:
            assert jnp.allclose(v, values_full[0])

        # Now call with offset
        value_offset = nll.value()
        assert value_offset == pytest.approx(5000.0)

        # Full value should still be the same
        value_full_after = nll.value(full=True)
        assert jnp.allclose(value_full_after, values_full[0])

    def test_full_parameter_with_params(self):
        """Test full parameter with parameter updates."""

        class ParamDist:
            def __init__(self, loc):
                self.base_loc = loc

            def logpdf(self, x, params=None):
                loc = self.base_loc
                if params and "loc" in params:
                    loc = params["loc"]
                return scipy_stats.norm(loc=loc, scale=1).logpdf(x)

        dist = ParamDist(0.0)
        data = np.array([0.0, 1.0, -1.0])

        options = NLLOptions.mean(start_value=1000)
        nll = NLL(dist, data, options=options)

        # Full value with default params
        value_full1 = nll.value(full=True)
        expected1 = -np.sum(scipy_stats.norm(0, 1).logpdf(data))
        assert value_full1 == pytest.approx(expected1)

        # Full value with different params
        value_full2 = nll.value(params={"loc": 0.5}, full=True)
        expected2 = -np.sum(scipy_stats.norm(0.5, 1).logpdf(data))
        assert value_full2 == pytest.approx(expected2)

        # Values should be different
        assert value_full1 != value_full2
