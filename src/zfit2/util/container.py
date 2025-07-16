"""Container utilities for zfit2.

This module provides helper functions for working with containers and collections,
including conversion and type checking utilities.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Container, Iterable
from typing import Any, TypeVar

import jax.numpy as jnp
import numpy as np

T = TypeVar("T")

# Default types to exclude from being treated as containers
DEFAULT_EXCLUDE_TYPES = (str, bytes, bytearray, np.ndarray, jnp.ndarray)


def is_container(
    obj: Any,
    *,
    exclude_types: tuple[type, ...] | None = None,
    require_iterable: bool = True,
) -> bool:
    """Check if an object is a container.

    An object is considered a container if it is a Collection (has __len__ and __iter__)
    or an Iterable (has __iter__), but not one of the excluded types.

    Args:
        obj: The object to check.
        exclude_types: Types to exclude from being considered containers.
            If None, uses DEFAULT_EXCLUDE_TYPES (str, bytes, bytearray, ndarray).
        require_iterable: If True (default), only considers objects that are iterable.
            If False, any object with __contains__ is considered a container.

    Returns:
        True if the object is a container, False otherwise.

    Examples:
        >>> is_container([1, 2, 3])
        True
        >>> is_container((1, 2, 3))
        True
        >>> is_container({1, 2, 3})
        True
        >>> is_container("hello")  # strings are excluded by default
        False
        >>> is_container(np.array([1, 2, 3]))  # arrays are excluded by default
        False
        >>> is_container("hello", exclude_types=())  # include strings
        True
        >>> is_container(42)
        False
    """
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_TYPES

    # Check if it's an excluded type
    if isinstance(obj, exclude_types):
        return False

    # Check if it's a container based on the require_iterable flag
    if require_iterable:
        # Must be iterable (Collection is a subclass of Iterable)
        return isinstance(obj, Collection | Iterable)
    else:
        # Just needs __contains__ method
        return hasattr(obj, "__contains__")


def to_container(
    obj: Any,
    *,
    container_type: type[Container[T]] | Callable[[Iterable[T]], Container[T]] = tuple,
    exclude_types: tuple[type, ...] | None = None,
    force: bool = False,
) -> Container[T]:
    """Convert an object to a container.

    If the object is already a container (and not an excluded type), it is returned as-is
    unless force=True. Otherwise, it is wrapped in the specified container type.

    Args:
        obj: The object to convert to a container.
        container_type: The container type to use. Can be a type (e.g., list, tuple, set)
            or a callable that takes an iterable and returns a container.
            Defaults to tuple.
        exclude_types: Types to exclude from being considered containers.
            If None, uses DEFAULT_EXCLUDE_TYPES (str, bytes, bytearray, ndarray).
        force: If True, always converts to the specified container_type even if
            the object is already a container.

    Returns:
        A container of the specified type containing the object(s).

    Raises:
        TypeError: If container_type is not callable.

    Examples:
        >>> to_container(5)
        (5,)
        >>> to_container([1, 2, 3])
        [1, 2, 3]
        >>> to_container([1, 2, 3], force=True)
        (1, 2, 3)
        >>> to_container([1, 2, 3], container_type=list, force=True)
        [1, 2, 3]
        >>> to_container("hello")  # strings are wrapped
        ('hello',)
        >>> to_container("hello", exclude_types=())  # strings not excluded
        'hello'
        >>> to_container(np.array([1, 2, 3]))  # arrays are wrapped
        (array([1, 2, 3]),)
        >>> to_container({1, 2, 3}, container_type=list, force=True)
        [1, 2, 3]
    """
    if not callable(container_type):
        msg = f"container_type must be callable, got {type(container_type).__name__}"
        raise TypeError(msg)

    # If it's already a container and we're not forcing conversion
    if not force and is_container(obj, exclude_types=exclude_types):
        return obj

    # Convert to container
    # Check if we should treat it as a single item (excluded types or not iterable)
    if exclude_types is None:
        exclude_types = DEFAULT_EXCLUDE_TYPES

    if isinstance(obj, exclude_types) or not is_container(obj, exclude_types=()):
        # It's an excluded type or not iterable, wrap in a single-element container
        return container_type([obj])
    else:
        # It's an iterable and not excluded, so we can pass it directly to container_type
        return container_type(obj)
