"""Context managers and decorators for backend operations."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

# Import types for type checking only
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    pass

# Type variables
T = TypeVar("T")
R = TypeVar("R")


class use_backend:
    """Context manager for temporarily switching backends.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.context import use_backend
        >>>
        >>> # Use the default backend
        >>> x = znp.array([1, 2, 3])
        >>>
        >>> # Temporarily use the NumPy backend
        >>> with use_backend("numpy"):
        ...     y = znp.mean(x)
        >>>
        >>> # Back to the default backend
        >>> result = znp.sum(x)
    """

    def __init__(self, backend_name: Literal["jax", "numpy", "sympy"]):
        """Initialize the context manager.

        Args:
            backend_name: The name of the backend to use.
        """
        self.backend_name = backend_name
        self._prev_backend_name = None

    def __enter__(self):
        """Enter the context and switch to the specified backend."""
        from . import _CURRENT_BACKEND_NAME, get_backend

        # Save the current backend name
        self._prev_backend_name = _CURRENT_BACKEND_NAME

        # Switch to the new backend
        from . import set_backend

        try:
            set_backend(self.backend_name)
        except ImportError as e:
            # If the requested backend is not available, raise a more informative error
            raise ImportError(f"Cannot switch to backend '{self.backend_name}': {e}")

        # Return the backend instance
        return get_backend()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore the previous backend."""
        from . import set_backend

        # Restore the previous backend
        if self._prev_backend_name is not None:
            set_backend(self._prev_backend_name)

        # Don't suppress exceptions
        return False


def host_callback(
    host_function: Callable[[T], R],
) -> Callable[[Callable[..., T]], Callable[..., R]]:
    """Decorator to call a host function from a backend computation.

    This decorator allows you to execute a Python function on the host
    CPU during a backend computation. This is useful for debugging,
    logging, or interacting with external systems that aren't supported
    by the backend.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.context import host_callback
        >>>
        >>> # Define a host function that uses NumPy
        >>> def numpy_function(x):
        ...     import numpy as np
        ...     return np.mean(x)
        >>>
        >>> # Use host callback in JAX computation
        >>> @host_callback(numpy_function)
        ... def compute_mean(x):
        ...     return x
        >>>
        >>> # JAX will call numpy_function on the host
        >>> result = compute_mean(znp.array([1, 2, 3, 4]))

    Args:
        host_function: A Python function to call on the host.

    Returns:
        A decorator that transforms a function to use host_callback.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from . import get_backend

            backend = get_backend()

            # Compute the argument to pass to the host function
            arg = func(*args, **kwargs)

            # If we're using JAX, use its host_callback mechanism
            if backend.name == "JAX":
                try:
                    return backend.host_callback(host_function, arg)
                except Exception as e:
                    warnings.warn(
                        f"JAX host_callback failed: {e}. Falling back to direct call."
                    )
                    return host_function(arg)
            else:
                # For other backends, just call the host function directly
                return host_function(arg)

        return wrapper

    return decorator


def numpy_fallback(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to automatically fall back to NumPy if a function fails in the current backend.

    This is useful for operations that may not be supported in all backends,
    especially when using experimental features or custom operations.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.context import numpy_fallback
        >>>
        >>> # Automatically fall back to NumPy if function fails
        >>> @numpy_fallback
        ... def complex_operation(x):
        ...     # This might fail in some backends
        ...     return z.backend.some_advanced_operation(x)

    Args:
        func: The function to wrap with fallback behavior.

    Returns:
        A wrapped function that falls back to NumPy if the original function fails.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from . import get_backend

        current_backend = get_backend()

        try:
            # Try with the current backend
            return func(*args, **kwargs)
        except Exception as e:
            # If current backend is already NumPy, just re-raise
            if current_backend.name == "NumPy":
                raise

            # Log a warning
            warnings.warn(
                f"Operation failed in {current_backend.name} backend: {e}. "
                f"Falling back to NumPy backend."
            )

            # Switch to NumPy backend temporarily
            with use_backend("numpy"):
                return func(*args, **kwargs)

    return wrapper
