"""Tests for statistic options."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from zfit2.statistic.options import NLLOptions, NLLOptionsLegacy


class TestNLLOptions:
    """Test NLLOptions class."""

    def test_basic_offset_configuration(self):
        """Test basic offset configuration."""
        # Test different offset methods
        options_mean = NLLOptions().offset("mean", start_value=10000)
        config = options_mean.get_offset_config()
        assert config["method"] == "mean"
        assert jnp.allclose(config["start_value"], 10000)

        options_median = NLLOptions().offset("median", start_value=5000)
        config = options_median.get_offset_config()
        assert config["method"] == "median"
        assert jnp.allclose(config["start_value"], 5000)

        options_none = NLLOptions().offset("none")
        config = options_none.get_offset_config()
        assert config["method"] == "none"
        assert jnp.allclose(config["start_value"], 0.0)

        options_elementwise = NLLOptions().offset("elementwise", start_value=2000)
        config = options_elementwise.get_offset_config()
        assert config["method"] == "elementwise"
        assert jnp.allclose(config["start_value"], 2000)

    def test_custom_offset_function(self):
        """Test custom offset function configuration."""

        def custom_offset(values):
            return jnp.percentile(values, 90)

        options = NLLOptions().offset(custom_offset)
        config = options.get_offset_config()
        assert config["method"] == "custom"
        assert config["function"] is custom_offset

    def test_sum_configuration(self):
        """Test sum method configuration."""
        options = NLLOptions().sum("standard")
        assert options.get_sum_config() == {"method": "standard"}

    def test_chaining(self):
        """Test method chaining."""
        options = NLLOptions().offset("mean", start_value=10000).sum("standard")

        config = options.get_offset_config()
        assert config["method"] == "mean"
        assert jnp.allclose(config["start_value"], 10000)
        assert options.get_sum_config() == {"method": "standard"}

    def test_double_configuration_error(self):
        """Test that configuring the same method twice raises error."""
        options = NLLOptions().offset("mean")

        # Should raise error when trying to configure offset again
        with pytest.raises(ValueError, match="Offset method already configured"):
            options.offset("median")

        # Same for sum
        options2 = NLLOptions().sum("standard")
        with pytest.raises(ValueError, match="Sum method already configured"):
            options2.sum("standard")

    def test_force_reconfiguration(self):
        """Test forced reconfiguration."""
        options = (
            NLLOptions()
            .offset("mean", start_value=1000)
            .offset("median", start_value=2000, force=True)
        )

        config = options.get_offset_config()
        assert config["method"] == "median"
        assert jnp.allclose(config["start_value"], 2000)

        # Test with sum
        options2 = NLLOptions().sum("standard").sum("standard", force=True)

        assert options2.get_sum_config() == {"method": "standard"}

    def test_invalid_methods(self):
        """Test invalid method names raise errors."""
        with pytest.raises(ValueError, match="Invalid offset method: invalid"):
            NLLOptions().offset("invalid")

        with pytest.raises(ValueError, match="Invalid sum method: invalid"):
            NLLOptions().sum("invalid")

    def test_immutability(self):
        """Test that options are immutable."""
        options1 = NLLOptions()
        options2 = options1.offset("mean")
        options3 = options2.sum("standard")

        # Each should be a different instance
        assert options1 is not options2
        assert options2 is not options3

        # Original should be unchanged
        assert options1.get_offset_config() is None
        assert options1.get_sum_config() is None

        # options2 should have offset but not sum
        assert options2.get_offset_config() is not None
        assert options2.get_sum_config() is None

        # options3 should have both
        assert options3.get_offset_config() is not None
        assert options3.get_sum_config() is not None

    def test_repr(self):
        """Test string representation."""
        options1 = NLLOptions()
        assert repr(options1) == "NLLOptions()"

        options2 = NLLOptions().offset("mean")
        assert repr(options2) == "NLLOptions(offset=mean, start_value=0)"

        options3 = NLLOptions().offset("mean", start_value=5000)
        assert repr(options3) == "NLLOptions(offset=mean, start_value=5000)"

        options4 = NLLOptions().offset("mean", start_value=10000).sum("standard")
        assert (
            repr(options4) == "NLLOptions(offset=mean, start_value=10000, sum=standard)"
        )

        # Custom function
        options5 = NLLOptions().offset(lambda x: x)
        assert repr(options5) == "NLLOptions(offset=custom)"

    def test_factory_methods(self):
        """Test convenience factory methods."""
        # Default
        options = NLLOptions.default()
        config = options.get_offset_config()
        assert config["method"] == "mean"
        assert jnp.allclose(config["start_value"], 10000.0)

        # None
        options = NLLOptions.none()
        config = options.get_offset_config()
        assert config["method"] == "none"
        assert jnp.allclose(config["start_value"], 0.0)

        # Mean
        options = NLLOptions.mean()
        config = options.get_offset_config()
        assert config["method"] == "mean"
        assert jnp.allclose(config["start_value"], 10000.0)

        options = NLLOptions.mean(5000)
        config = options.get_offset_config()
        assert config["method"] == "mean"
        assert jnp.allclose(config["start_value"], 5000.0)

        # Median
        options = NLLOptions.median()
        config = options.get_offset_config()
        assert config["method"] == "median"
        assert jnp.allclose(config["start_value"], 10000.0)

        # Elementwise
        options = NLLOptions.elementwise(2000)
        config = options.get_offset_config()
        assert config["method"] == "elementwise"
        assert jnp.allclose(config["start_value"], 2000.0)


class TestNLLOptionsCompatibility:
    """Test NLLOptions compatibility wrapper."""

    def test_basic_creation(self):
        """Test basic NLLOptions creation using legacy interface."""
        options = NLLOptionsLegacy("mean", 1000)
        assert options.get_offset_method() == "mean"
        assert options.start_value == 1000
        assert options.get_offset_method() == "mean"

    def test_colon_format(self):
        """Test legacy colon format."""
        options = NLLOptionsLegacy("mean:5000")
        assert options.get_offset_method() == "mean"
        assert options.start_value == 5000.0

        options2 = NLLOptionsLegacy("median:2000.5")
        assert options2.get_offset_method() == "median"
        assert options2.start_value == 2000.5

        # Invalid value after colon
        with pytest.raises(ValueError, match="Invalid start value"):
            NLLOptionsLegacy("mean:invalid")

    def test_custom_function(self):
        """Test custom function in NLLOptions."""

        def custom_fn(x):
            return x * 2

        options = NLLOptionsLegacy(custom_fn)
        assert options.get_offset_method() == "custom"
        assert options.get_offset_method() == "custom"

    def test_factory_methods(self):
        """Test NLLOptions factory methods."""
        options_none = NLLOptions.none()
        assert options_none.get_offset_method() == "none"

        options_mean = NLLOptions.mean(5000)
        assert options_mean.get_offset_method() == "mean"
        assert options_mean.start_value == 5000

        options_median = NLLOptions.median(3000)
        assert options_median.get_offset_method() == "median"
        assert options_median.start_value == 3000

        options_elementwise = NLLOptions.elementwise(1000)
        assert options_elementwise.get_offset_method() == "elementwise"
        assert options_elementwise.start_value == 1000

        def custom(x):
            return x

        options_custom = NLLOptions.custom(custom)
        assert options_custom.get_offset_method() == "custom"


class TestNLLOptionsIntegration:
    """Integration tests with actual usage patterns."""

    def test_complex_chaining(self):
        """Test complex chaining scenarios."""
        # Build options step by step
        options = NLLOptions()
        options = options.offset("mean", start_value=1000)

        # Try to add offset again - should fail
        with pytest.raises(ValueError):
            options.offset("median")

        # Add sum configuration
        options = options.sum("standard")

        # Verify final state
        assert options.get_offset_config()["method"] == "mean"
        assert options.get_sum_config()["method"] == "standard"

    def test_options_independence(self):
        """Test that different option instances are independent."""
        base = NLLOptions()

        opt1 = base.offset("mean", start_value=1000)
        opt2 = base.offset("median", start_value=2000)

        # Both should work independently
        assert opt1.get_offset_config()["method"] == "mean"
        assert jnp.allclose(opt1.get_offset_config()["start_value"], 1000)

        assert opt2.get_offset_config()["method"] == "median"
        assert jnp.allclose(opt2.get_offset_config()["start_value"], 2000)

        # Base should remain unchanged
        assert base.get_offset_config() is None
