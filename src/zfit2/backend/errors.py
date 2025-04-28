"""Errors for the backend module."""

from __future__ import annotations


class BackendError(Exception):
    """Base class for all backend-related errors."""


class NotImplementedInBackend(BackendError):
    """Error raised when a function is not implemented in a backend."""

    def __init__(self, func_name: str, backend_name: str):
        """Initialize the error.

        Args:
            func_name: The name of the function that is not implemented.
            backend_name: The name of the backend that doesn't implement the function.
        """
        self.func_name = func_name
        self.backend_name = backend_name
        message = (
            f"Function '{func_name}' is not implemented in the {backend_name} backend."
        )
        super().__init__(message)
