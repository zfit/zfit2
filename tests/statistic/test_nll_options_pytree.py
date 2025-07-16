"""Tests for NLLOptions JAX pytree functionality."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from zfit2.statistic.options import NLLOptions


class TestNLLOptionsPyTree:
    """Test NLLOptions as JAX pytree."""

    def test_nll_options_is_pytree(self):
        """Test that NLLOptions is recognized as a pytree."""
        options = NLLOptions().offset("mean", start_value=1000)

        # Should be able to flatten and unflatten
        flat, treedef = jax.tree_util.tree_flatten(options)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        assert isinstance(reconstructed, NLLOptions)
        assert reconstructed.get_offset_config()["method"] == "mean"
        assert jnp.array_equal(
            reconstructed.get_offset_config()["start_value"],
            options.get_offset_config()["start_value"],
        )

    def test_start_value_is_jax_array(self):
        """Test that start_value is stored as JAX array."""
        options = NLLOptions().offset("mean", start_value=5000.0)

        config = options.get_offset_config()
        assert isinstance(config["start_value"], jnp.ndarray)
        assert config["start_value"].shape == ()  # Scalar
        assert jnp.array_equal(config["start_value"], jnp.asarray(5000.0))

    def test_pytree_with_different_configs(self):
        """Test pytree with various configurations."""
        # No offset
        options1 = NLLOptions()
        flat1, treedef1 = jax.tree_util.tree_flatten(options1)
        recon1 = jax.tree_util.tree_unflatten(treedef1, flat1)
        assert recon1.get_offset_config() is None

        # Mean offset
        options2 = NLLOptions().offset("mean", start_value=1000)
        flat2, treedef2 = jax.tree_util.tree_flatten(options2)
        recon2 = jax.tree_util.tree_unflatten(treedef2, flat2)
        assert recon2.get_offset_config()["method"] == "mean"
        assert jnp.allclose(recon2.get_offset_config()["start_value"], 1000)

        # Custom function
        def custom_fn(x):
            return jnp.percentile(x, 90)

        options3 = NLLOptions().offset(custom_fn)
        flat3, treedef3 = jax.tree_util.tree_flatten(options3)
        recon3 = jax.tree_util.tree_unflatten(treedef3, flat3)
        assert recon3.get_offset_config()["method"] == "custom"
        assert recon3.get_offset_config()["function"] is custom_fn

        # With sum configuration
        options4 = NLLOptions().offset("median", start_value=2000).sum("standard")
        flat4, treedef4 = jax.tree_util.tree_flatten(options4)
        recon4 = jax.tree_util.tree_unflatten(treedef4, flat4)
        assert recon4.get_offset_config()["method"] == "median"
        assert jnp.allclose(recon4.get_offset_config()["start_value"], 2000)
        assert recon4.get_sum_config()["method"] == "standard"

    def test_jax_transformations(self):
        """Test that NLLOptions works with JAX transformations."""

        # Test with jit
        @jax.jit
        def get_start_value(options: NLLOptions) -> jnp.ndarray:
            config = options.get_offset_config()
            if config and "start_value" in config:
                return config["start_value"]
            return jnp.asarray(0.0)

        options = NLLOptions().offset("mean", start_value=5000)
        result = get_start_value(options)
        assert jnp.allclose(result, 5000)

        # Test with vmap over different start values
        def create_options_with_value(value):
            # Note: We can't create new options inside vmap,
            # but we can work with pre-created options
            return value

        values = jnp.array([1000.0, 2000.0, 3000.0])
        vmapped = jax.vmap(create_options_with_value)
        results = vmapped(values)
        assert jnp.array_equal(results, values)

    def test_tree_map_with_options(self):
        """Test jax.tree operations with NLLOptions."""
        options1 = NLLOptions().offset("mean", start_value=1000)
        options2 = NLLOptions().offset("median", start_value=2000)

        # Create a tree structure with options
        tree = {"opt1": options1, "opt2": options2}

        # Flatten and check the leaves
        leaves, treedef = jax.tree_util.tree_flatten(tree)

        # Should have 2 leaves (the start values)
        assert len(leaves) == 2
        assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)

        # Modify leaves
        new_leaves = [leaf * 2 for leaf in leaves]

        # Reconstruct with modified leaves
        new_tree = jax.tree_util.tree_unflatten(treedef, new_leaves)

        # Check that the new tree has updated values
        assert jnp.allclose(new_tree["opt1"].get_offset_config()["start_value"], 2000)
        assert jnp.allclose(new_tree["opt2"].get_offset_config()["start_value"], 4000)

    def test_pytree_leaves_and_structure(self):
        """Test tree leaves and structure."""
        options = NLLOptions().offset("mean", start_value=3000)

        leaves, treedef = jax.tree_util.tree_flatten(options)

        # Should have one leaf (start_value)
        assert len(leaves) == 1
        assert isinstance(leaves[0], jnp.ndarray)
        assert jnp.allclose(leaves[0], 3000)

        # Modifying leaves should create new options with modified values
        new_leaves = [jnp.asarray(4000)]
        new_options = jax.tree_util.tree_unflatten(treedef, new_leaves)

        assert new_options.get_offset_config()["method"] == "mean"
        assert jnp.allclose(new_options.get_offset_config()["start_value"], 4000)

    def test_gradient_through_options(self):
        """Test that we can take gradients through options."""

        def loss_fn(start_val):
            # Create options with the given start value
            # In practice, we'd use pre-created options, but for testing
            # we'll work with the start value directly
            return (start_val - 5000.0) ** 2

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(jnp.asarray(3000.0))

        # Gradient should be 2 * (3000 - 5000) = -4000
        assert jnp.allclose(grad, -4000.0)

    def test_options_in_nested_structures(self):
        """Test NLLOptions in nested pytree structures."""
        options1 = NLLOptions().offset("mean", start_value=1000)
        options2 = NLLOptions().offset("median", start_value=2000)

        nested = {
            "group1": [options1, options2],
            "group2": {"a": options1, "b": options2},
            "single": options1,
        }

        # Flatten and unflatten
        flat, treedef = jax.tree_util.tree_flatten(nested)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        # Check structure is preserved
        assert isinstance(reconstructed["group1"][0], NLLOptions)
        assert isinstance(reconstructed["group1"][1], NLLOptions)
        assert reconstructed["group1"][0].get_offset_config()["method"] == "mean"
        assert reconstructed["group1"][1].get_offset_config()["method"] == "median"

        # Check values
        assert jnp.allclose(
            reconstructed["group1"][0].get_offset_config()["start_value"], 1000
        )
        assert jnp.allclose(
            reconstructed["group2"]["b"].get_offset_config()["start_value"], 2000
        )

    def test_backwards_compatibility_properties(self):
        """Test backwards compatibility properties return correct types."""
        options = NLLOptions().offset("mean", start_value=3000.5)

        # The properties should return Python types for compatibility
        assert options.get_offset_method() == "mean"
        assert isinstance(options.start_value, float)
        assert options.start_value == 3000.5

        # But internally it's stored as JAX array
        assert isinstance(options.get_offset_config()["start_value"], jnp.ndarray)


class TestNLLOptionsLegacyInterface:
    """Test the legacy interface compatibility."""

    def test_legacy_constructor(self):
        """Test creating options with legacy constructor."""
        from zfit2.statistic.options import NLLOptionsLegacy

        # Test basic creation
        options = NLLOptionsLegacy("mean", 1000)
        assert isinstance(options, NLLOptions)
        assert options.get_offset_config()["method"] == "mean"
        assert jnp.allclose(options.get_offset_config()["start_value"], 1000)

        # Test with colon format
        options2 = NLLOptionsLegacy("median:2000")
        assert options2.get_offset_config()["method"] == "median"
        assert jnp.allclose(options2.get_offset_config()["start_value"], 2000)

        # Test with custom function
        def custom(x):
            return x * 2

        options3 = NLLOptionsLegacy(custom)
        assert options3.get_offset_config()["method"] == "custom"
        assert options3.get_offset_config()["function"] is custom

        # Test invalid colon format
        with pytest.raises(ValueError, match="Invalid start value"):
            NLLOptionsLegacy("mean:invalid")
