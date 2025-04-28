"""Backend module for scientific computing.

This module provides a unified interface to different computational backends:
- JAX: Accelerated array computing and automatic differentiation
- NumPy/SciPy: Standard scientific computing
- SymPy: Symbolic mathematics
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from .base import BackendBase
from .context import host_callback, use_backend
from .errors import BackendError, NotImplementedInBackend

# Type variables for PyTree operations
T = TypeVar("T")
U = TypeVar("U")

# Backend instances - initialized on first access
_JAX_BACKEND = None
_NUMPY_BACKEND = None
_SYMPY_BACKEND = None
_CURRENT_BACKEND_NAME = None


def get_backend(name: Optional[Literal["jax", "numpy", "sympy"]] = None) -> BackendBase:
    """Get the backend with the given name.

    Args:
        name: The name of the backend to get. If None, returns the current backend.

    Returns:
        The requested backend instance.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    global _JAX_BACKEND, _NUMPY_BACKEND, _SYMPY_BACKEND, _CURRENT_BACKEND_NAME

    if name is None:
        # Use the current backend if set, otherwise initialize with a default
        if _CURRENT_BACKEND_NAME is None:
            # Default to JAX if available, otherwise NumPy
            try:
                return get_backend("jax")
            except (ImportError, BackendError):
                return get_backend("numpy")
        else:
            return get_backend(_CURRENT_BACKEND_NAME)

    name = name.lower()
    if name == "jax":
        if _JAX_BACKEND is None:
            try:
                from .jax_backend import JAXBackend

                _JAX_BACKEND = JAXBackend()
            except ImportError:
                raise ImportError(
                    "JAX is not installed. Please install it with `pip install jax`."
                )
        _CURRENT_BACKEND_NAME = name
        return _JAX_BACKEND

    elif name == "numpy":
        if _NUMPY_BACKEND is None:
            from .numpy_backend import NumPyBackend

            _NUMPY_BACKEND = NumPyBackend()
        _CURRENT_BACKEND_NAME = name
        return _NUMPY_BACKEND

    elif name == "sympy":
        if _SYMPY_BACKEND is None:
            try:
                from .sympy_backend import SymPyBackend

                _SYMPY_BACKEND = SymPyBackend()
            except ImportError:
                raise ImportError(
                    "SymPy is not installed. Please install it with `pip install sympy`."
                )
        _CURRENT_BACKEND_NAME = name
        return _SYMPY_BACKEND

    else:
        raise ValueError(
            f"Unsupported backend: {name}. Available backends: jax, numpy, sympy"
        )


def set_backend(name: Literal["jax", "numpy", "sympy"]) -> None:
    """Set the active backend.

    Args:
        name: The name of the backend to set as active.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    global _CURRENT_BACKEND_NAME
    # Get the backend to initialize it and validate
    get_backend(name)
    _CURRENT_BACKEND_NAME = name.lower()


class BackendInterface:
    """Interface to the active backend's transformations and utilities."""

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the active backend."""
        return getattr(get_backend(), name)


# Initialize backend from environment variable if set
if "ZFIT_BACKEND" in os.environ:
    try:
        set_backend(os.environ["ZFIT_BACKEND"].lower())
    except (ImportError, ValueError, BackendError) as e:
        import warnings

        warnings.warn(
            f"Failed to set backend from ZFIT_BACKEND environment variable: {e}"
        )


# Expose the interface
backend = BackendInterface()

# Expose common functions for direct imports
from .optimize import curve_fit, minimize, root
from .vectorize import vmap

__all__ = [
    # Main interface
    "backend",
    # Backend management
    "get_backend",
    "set_backend",
    "use_backend",
    "host_callback",
    "BackendError",
    "NotImplementedInBackend",
    # Common utilities
    "vmap",
    "minimize",
    "root",
    "curve_fit",
]
