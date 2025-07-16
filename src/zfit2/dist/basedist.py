"""Base distribution class for zfit2.

This module provides the abstract base class for all distributions
in zfit2, defining the common interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import jax.numpy as jnp


class BaseDist(ABC):
    """Abstract base class for distributions.

    This class defines the interface that all distributions must implement,
    providing methods for probability density evaluation, sampling, etc.
    """

    def logpdf(self, x: Any, params: dict | None = None) -> jnp.ndarray:
        """Compute log probability density with preprocessing.

        Args:
            x: Data points to evaluate
            params: Optional parameter dictionary

        Returns:
            Log probability density values
        """
        # Ensure x is a JAX array
        x = jnp.asarray(x)

        # Call the implementation
        return self._logpdf(x, params)

    @abstractmethod
    def _logpdf(self, x: jnp.ndarray, params: dict | None = None) -> jnp.ndarray:
        """Private method to compute log probability density.

        Args:
            x: Data points to evaluate (guaranteed to be JAX array)
            params: Optional parameter dictionary

        Returns:
            Log probability density values
        """
        raise NotImplementedError("Subclasses must implement _logpdf()")

    def pdf(self, x: Any, params: dict | None = None) -> jnp.ndarray:
        """Compute probability density.

        Args:
            x: Data points to evaluate
            params: Optional parameter dictionary

        Returns:
            Probability density values
        """
        return jnp.exp(self.logpdf(x, params))

    @abstractmethod
    def sample(self, size: int, params: dict | None = None) -> jnp.ndarray:
        """Generate random samples.

        Args:
            size: Number of samples to generate
            params: Optional parameter dictionary

        Returns:
            Random samples as JAX array
        """
        raise NotImplementedError("Subclasses must implement sample()")
