"""Tests for collection utility functions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from zfit2.util import DEFAULT_EXCLUDE_TYPES, is_collection, to_collection


class TestIsCollection:
    """Test is_collection function."""

    def test_is_collection_basic_types(self):
        """Test is_collection with basic Python types."""
        # Collections
        assert is_collection([1, 2, 3]) is True
        assert is_collection((1, 2, 3)) is True
        assert is_collection({1, 2, 3}) is True
        assert is_collection({1: "a", 2: "b"}) is True
        assert is_collection(range(5)) is True

        # Not collections
        assert is_collection(42) is False
        assert is_collection(3.14) is False
        assert is_collection(True) is False
        assert is_collection(None) is False

    def test_is_collection_excluded_types(self):
        """Test that excluded types are not considered collections."""
        # Default exclusions
        assert is_collection("hello") is False
        assert is_collection(b"bytes") is False
        assert is_collection(bytearray(b"bytearray")) is False
        assert is_collection(np.array([1, 2, 3])) is False
        assert is_collection(jnp.array([1, 2, 3])) is False

    def test_is_collection_custom_exclude_types(self):
        """Test is_collection with custom exclude_types."""
        # Empty exclusions - strings become collections
        assert is_collection("hello", exclude_types=()) is True
        assert is_collection(b"bytes", exclude_types=()) is True

        # Custom exclusions
        assert is_collection([1, 2, 3], exclude_types=(list,)) is False
        assert is_collection((1, 2, 3), exclude_types=(tuple,)) is False

    def test_is_collection_generators(self):
        """Test is_collection with generator expressions."""
        gen = (x for x in range(3))
        assert is_collection(gen) is True

    def test_is_collection_custom_classes(self):
        """Test is_collection with custom classes."""

        class CustomIterable:
            def __iter__(self):
                return iter([1, 2, 3])

        class CustomCollection:
            def __iter__(self):
                return iter([1, 2, 3])

            def __len__(self):
                return 3

        class NotIterable:
            pass

        assert is_collection(CustomIterable()) is True
        assert is_collection(CustomCollection()) is True
        assert is_collection(NotIterable()) is False

    def test_is_collection_require_iterable_false(self):
        """Test is_collection with require_iterable=False."""

        class HasContains:
            def __contains__(self, item):
                return item in [1, 2, 3]

        obj = HasContains()
        assert is_collection(obj, require_iterable=False) is True
        assert is_collection(obj, require_iterable=True) is False


class TestToCollection:
    """Test to_collection function."""

    def test_to_collection_non_collections(self):
        """Test to_collection with non-collection inputs."""
        # Single values get wrapped
        assert to_collection(42) == (42,)
        assert to_collection(3.14) == (3.14,)
        assert to_collection(True) == (True,)
        assert to_collection(None) == (None,)

    def test_to_collection_excluded_types(self):
        """Test to_collection with excluded types."""
        # Default exclusions get wrapped
        assert to_collection("hello") == ("hello",)
        assert to_collection(b"bytes") == (b"bytes",)
        assert to_collection(bytearray(b"bytearray")) == (bytearray(b"bytearray"),)

        arr = np.array([1, 2, 3])
        result = to_collection(arr)
        assert len(result) == 1
        assert np.array_equal(result[0], arr)

        jarr = jnp.array([1, 2, 3])
        result = to_collection(jarr)
        assert len(result) == 1
        assert jnp.array_equal(result[0], jarr)

    def test_to_collection_existing_collections(self):
        """Test to_collection with existing collections."""
        # Existing collections are returned as-is
        lst = [1, 2, 3]
        assert to_collection(lst) is lst

        tpl = (1, 2, 3)
        assert to_collection(tpl) is tpl

        st = {1, 2, 3}
        assert to_collection(st) is st

        dct = {1: "a", 2: "b"}
        assert to_collection(dct) is dct

    def test_to_collection_force(self):
        """Test to_collection with force=True."""
        # Force conversion to specified type
        lst = [1, 2, 3]
        assert to_collection(lst, force=True) == (1, 2, 3)
        assert to_collection(lst, collection_type=list, force=True) == [1, 2, 3]
        assert to_collection(lst, collection_type=set, force=True) == {1, 2, 3}

        # Force conversion of single values
        assert to_collection(42, collection_type=list) == [42]
        assert to_collection(42, collection_type=set) == {42}

    def test_to_collection_custom_collection_type(self):
        """Test to_collection with custom collection types."""
        # Different collection types
        assert to_collection(42, collection_type=list) == [42]
        assert to_collection(42, collection_type=set) == {42}
        assert to_collection(42, collection_type=frozenset) == frozenset({42})

        # With existing collections and force
        assert to_collection([1, 2, 3], collection_type=set, force=True) == {1, 2, 3}
        assert to_collection({1, 2, 3}, collection_type=list, force=True) == [1, 2, 3]

    def test_to_collection_custom_exclude_types(self):
        """Test to_collection with custom exclude_types."""
        # Empty exclusions - strings not wrapped
        assert to_collection("hello", exclude_types=()) == "hello"

        # Custom exclusions
        assert to_collection([1, 2, 3], exclude_types=(list,)) == ([1, 2, 3],)

    def test_to_collection_generator(self):
        """Test to_collection with generators."""
        gen = (x for x in range(3))
        result = to_collection(gen)
        # Generator is returned as-is since it's already a collection
        assert result is gen

        # With force=True, generator is consumed and converted
        gen2 = (x for x in range(3))
        result = to_collection(gen2, force=True)
        assert result == (0, 1, 2)

    def test_to_collection_invalid_collection_type(self):
        """Test to_collection with invalid collection_type."""
        with pytest.raises(TypeError, match="collection_type must be callable"):
            to_collection(42, collection_type="not callable")

    def test_to_collection_custom_callable(self):
        """Test to_collection with custom callable as collection_type."""

        def custom_collection(items):
            return f"Custom: {list(items)}"

        assert to_collection(42, collection_type=custom_collection) == "Custom: [42]"
        assert (
            to_collection([1, 2, 3], collection_type=custom_collection, force=True)
            == "Custom: [1, 2, 3]"
        )


class TestDefaultExcludeTypes:
    """Test DEFAULT_EXCLUDE_TYPES constant."""

    def test_default_exclude_types(self):
        """Test that DEFAULT_EXCLUDE_TYPES contains expected types."""
        assert str in DEFAULT_EXCLUDE_TYPES
        assert bytes in DEFAULT_EXCLUDE_TYPES
        assert bytearray in DEFAULT_EXCLUDE_TYPES
        assert np.ndarray in DEFAULT_EXCLUDE_TYPES
        assert jnp.ndarray in DEFAULT_EXCLUDE_TYPES
        assert len(DEFAULT_EXCLUDE_TYPES) == 5
