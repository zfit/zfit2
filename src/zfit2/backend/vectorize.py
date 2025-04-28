"""Vectorization utilities for the backend module."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TypeVar, Union

# Type variables
T = TypeVar("T")
R = TypeVar("R")


def vmap(fun: Callable, in_axes=0, out_axes=0) -> Callable:
    """Vectorize a function along specified axes.

    This is a convenience wrapper around the backend's vmap function that
    ensures the function is vectorized using the current active backend.

    Args:
        fun: The function to vectorize.
        in_axes: Specification of which axes to map over for each input.
        out_axes: Specification of which axes in the output correspond to mapped axes.

    Returns:
        A vectorized version of the function.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend import vectorize
        >>>
        >>> def f(x, y):
        ...     return x + y
        >>>
        >>> # Vectorize over the first axis of both inputs
        >>> vf = vectorize.vmap(f)
        >>>
        >>> # Apply to arrays
        >>> x = znp.array([[1, 2], [3, 4]])
        >>> y = znp.array([[10, 20], [30, 40]])
        >>> vf(x, y)  # Applies f to each row: [f([1, 2], [10, 20]), f([3, 4], [30, 40])]
    """
    from . import get_backend

    @functools.wraps(fun)
    def vmapped_fun(*args, **kwargs):
        # Get the current backend
        backend = get_backend()

        # Create the vectorized function using the current backend
        vectorized = backend.vmap(fun, in_axes=in_axes, out_axes=out_axes)

        # Apply it to the arguments
        return vectorized(*args, **kwargs)

    return vmapped_fun


def auto_batch(
    batch_dims: Union[int, Sequence[int]] = 0,
) -> Callable[[Callable], Callable]:
    """Decorator to automatically batch a function along specified dimensions.

    This decorator automatically applies vmap to a function, making it batch
    over the specified dimensions. It's a convenience wrapper for common vectorization
    patterns.

    Args:
        batch_dims: The dimensions to batch over. Can be an int (same dimension for all inputs)
                    or a sequence of integers (one per input).

    Returns:
        A decorator that applies vmap to the decorated function.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.vectorize import auto_batch
        >>>
        >>> # Automatically batch over the first dimension
        >>> @auto_batch(0)
        ... def add(x, y):
        ...     return x + y
        >>>
        >>> # Equivalent to:
        >>> # add = vmap(lambda x, y: x + y, in_axes=0, out_axes=0)
        >>>
        >>> # Apply to arrays
        >>> x = znp.array([[1, 2], [3, 4]])
        >>> y = znp.array([[10, 20], [30, 40]])
        >>> add(x, y)  # Returns array([[11, 22], [33, 44]])
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Handle kwargs by moving them to args
            # (vmap only works on positional arguments)
            if kwargs:
                # Get function signature
                import inspect

                sig = inspect.signature(func)

                # Create new args list with kwargs filled in
                new_args = list(args)
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        # Find the position of the parameter
                        param_idx = list(sig.parameters.keys()).index(param_name)

                        # Extend args if necessary
                        if len(new_args) <= param_idx:
                            new_args.extend([None] * (param_idx - len(new_args) + 1))

                        # Set the argument
                        new_args[param_idx] = kwargs[param_name]

                # Convert back to tuple
                args = tuple(new_args)

            # Determine in_axes
            if isinstance(batch_dims, int):
                in_axes = batch_dims
            elif len(batch_dims) < len(args):
                # Extend with the last value
                in_axes = list(batch_dims) + [batch_dims[-1]] * (
                    len(args) - len(batch_dims)
                )
            else:
                in_axes = batch_dims[: len(args)]

            # Apply vmap
            return vmap(func, in_axes=in_axes, out_axes=0)(*args)

        return wrapper

    return decorator
