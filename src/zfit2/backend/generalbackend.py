from __future__ import annotations


class Backend:
    """
    Backend for zfit using numpy.
    """

    def __init__(self):
        import numpy as np

        self._np = np
        self.backend = self._np

    def __repr__(self):
        return "NumpyLikeBackend"

    def __str__(self):
        return "NumpyLikeBackend"

    def __getattr__(self, name):
        """
        Get the attribute from the numpy module.
        """
        return getattr(self.backend, name)
