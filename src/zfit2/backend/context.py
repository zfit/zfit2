"""Context manager for temporarily changing the backend."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal, Optional


@contextmanager
def use_backend(name: Literal["jax", "numpy", "sympy"]):
    """Context manager for temporarily changing the backend.
    
    Args:
        name: The name of the backend to use temporarily.
        
    Example:
        >>> from zfit2.backend.context import use_backend
        >>> import zfit2.backend as zb
        >>> # Current backend
        >>> print(zb.get_backend().name)
        >>> # Temporarily use NumPy backend
        >>> with use_backend("numpy"):
        ...     print(zb.get_backend().name)
        ...     # Do computations with NumPy
        >>> # Back to original backend
        >>> print(zb.get_backend().name)
    """
    # Import here to avoid circular imports
    import zfit2.backend as zb
    
    # Save the current backend
    original_backend = zb.get_backend()
    try:
        # Set the requested backend
        zb.set_backend(name)
        # Yield control back to the caller
        yield
    finally:
        # Restore the original backend
        import sys
        sys.modules['zfit2.backend']._CURRENT_BACKEND = original_backend