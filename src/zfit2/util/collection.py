"""Collection utilities for zfit2.

This module provides helper functions for working with collections,
including conversion and type checking utilities.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Container, Iterable
from typing import Any, TypeVar

import jax.numpy as jnp
import numpy as np

T = TypeVar("T")

# Default types to exclude from being treated as collections
DEFAULT_EXCLUDE_TYPES = (str, bytes, bytearray, np.ndarray, jnp.ndarray)


def is_collection(
    obj: Any,
    *,
    exclude_types: tuple[type, ...] | None = None,
    require_iterable: bool = True,
) -> bool:
    """Check if an object is a collection.

    An object is considered a collection if it is a Collection (has __len__ and __iter__)
    or an Iterable (has __iter__), but not one of the excluded types.

    Args:
        obj: The object to check.
        exclude_types: Types to exclude from being considered collections.
            If None, uses DEFAULT_EXCLUDE_TYPES (str, bytes, bytearray, ndarray).
        require_iterable: If True (default), only considers objects that are iterable.
            If False, any object with __contains__ is considered a collection.

    Returns:
        True if the object is a collection, False otherwise.

    Examples:
        >>> is_collection([1, 2, 3])
        True
        >>> is_collection((1, 2, 3))
        True
        >>> is_collection({1, 2, 3})
        True
        >>> is_collection("hello")  # strings are excluded by default
        False
        >>> is_collection(np.array([1, 2, 3]))  # arrays are excluded by default
        False
        >>> is_collection("hello", exclude_types=())  # include strings
        True
        >>> is_collection(42)
        False
    """
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_TYPES

    # Check if it's an excluded type
    if isinstance(obj, exclude_types):
        return False

    # Check if it's a collection based on the require_iterable flag
    if require_iterable:
        # Must be iterable (Collection is a subclass of Iterable)
        return isinstance(obj, Collection | Iterable)
    else:
        # Just needs __contains__ method
        return hasattr(obj, "__contains__")


def to_collection(
    obj: Any,
    *,
    collection_type: type[Container[T]] | Callable[[Iterable[T]], Container[T]] = tuple,
    exclude_types: tuple[type, ...] | None = None,
    force: bool = False,
) -> Container[T]:
    """Convert an object to a collection.

    If the object is already a collection (and not an excluded type), it is returned as-is
    unless force=True. Otherwise, it is wrapped in the specified collection type.

    Args:
        obj: The object to convert to a collection.
        collection_type: The collection type to use. Can be a type (e.g., list, tuple, set)
            or a callable that takes an iterable and returns a collection.
            Defaults to tuple.
        exclude_types: Types to exclude from being considered collections.
            If None, uses DEFAULT_EXCLUDE_TYPES (str, bytes, bytearray, ndarray).
        force: If True, always converts to the specified collection_type even if
            the object is already a collection.

    Returns:
        A collection of the specified type containing the object(s).

    Raises:
        TypeError: If collection_type is not callable.

    Examples:
        >>> to_collection(5)
        (5,)
        >>> to_collection([1, 2, 3])
        [1, 2, 3]
        >>> to_collection([1, 2, 3], force=True)
        (1, 2, 3)
        >>> to_collection([1, 2, 3], collection_type=list, force=True)
        [1, 2, 3]
        >>> to_collection("hello")  # strings are wrapped
        ('hello',)
        >>> to_collection("hello", exclude_types=())  # strings not excluded
        'hello'
        >>> to_collection(np.array([1, 2, 3]))  # arrays are wrapped
        (array([1, 2, 3]),)
        >>> to_collection({1, 2, 3}, collection_type=list, force=True)
        [1, 2, 3]
    """
    if not callable(collection_type):
        msg = f"collection_type must be callable, got {type(collection_type).__name__}"
        raise TypeError(msg)

    # If it's already a collection and we're not forcing conversion
    if not force and is_collection(obj, exclude_types=exclude_types):
        return obj

    # Convert to collection
    # Check if we should treat it as a single item (excluded types or not iterable)
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_TYPES

    if isinstance(obj, exclude_types) or not is_collection(obj, exclude_types=()):
        # It's an excluded type or not iterable, wrap in a single-element collection
        return collection_type([obj])
    else:
        # It's an iterable and not excluded, so we can pass it directly to collection_type
        return collection_type(obj)
