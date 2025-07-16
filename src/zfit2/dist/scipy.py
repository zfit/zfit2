"""Wrapper for scipy distributions to provide unified interface.

This module provides a wrapper class that adapts scipy distributions
to work with zfit2's parameter system.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from scipy import stats as scipy_stats

from .basedist import BaseDist


class ScipyDist(BaseDist):
    """Wrapper for scipy distributions with unified interface.

    This class wraps scipy.stats distributions to provide a unified
    logpdf(x, params) interface that can work with zfit2 parameters.

    Attributes:
        dist: The underlying scipy distribution
        param_names: Names of the distribution parameters
    """

    def __init__(self, dist: Any, **kwargs):
        """Initialize the scipy distribution wrapper.

        Args:
            dist: A scipy.stats distribution class (not instance)
            **kwargs: Initial parameter values for the distribution
        """
        # Store the distribution class and name
        self._dist_cls = dist
        self._dist_name = (
            dist.name
            if hasattr(dist, "name")
            else str(dist).split(".")[-1].rstrip("'>")
        )

        # Create frozen distribution with initial parameters
        self._dist = dist(**kwargs)

        # Check if it's a discrete distribution
        self._is_discrete = hasattr(self._dist, "pmf")

        # Store parameter names and values
        self._params = kwargs
        self.param_names = list(kwargs.keys())

    def _logpdf(self, x: jnp.ndarray, params: dict | None = None) -> jnp.ndarray:
        """Private method to compute log probability density.

        Args:
            x: Data points to evaluate (JAX array)
            params: Optional parameter dictionary. If provided, these
                   override the distribution's current parameters.

        Returns:
            Log probability density values
        """

        if params is not None:
            # Create new distribution with updated parameters
            # Only update parameters that are provided
            updated_params = self._params.copy()
            for key, value in params.items():
                if key in updated_params:
                    updated_params[key] = value
            dist = self._dist_cls(**updated_params)
        else:
            dist = self._dist

        # Compute log pdf/pmf based on distribution type
        if self._is_discrete:
            result = dist.logpmf(x)
        else:
            result = dist.logpdf(x)

        # Convert back to JAX array
        return jnp.asarray(result)

    def sample(self, size: int, params: dict | None = None) -> jnp.ndarray:
        """Generate random samples.

        Args:
            size: Number of samples to generate
            params: Optional parameter dictionary

        Returns:
            Random samples
        """
        if params is not None:
            updated_params = self._params.copy()
            for key, value in params.items():
                if key in updated_params:
                    updated_params[key] = value
            dist = self._dist_cls(**updated_params)
        else:
            dist = self._dist

        # Generate samples
        samples = dist.rvs(size=size)

        # Convert to JAX array
        return jnp.asarray(samples)

    @property
    def params(self) -> dict:
        """Get current parameter values."""
        return self._params.copy()

    def __repr__(self) -> str:
        """String representation."""
        param_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"ScipyDist({self._dist_name}({param_str}))"


# Convenience functions to create common distributions
def Normal(loc: float = 0.0, scale: float = 1.0) -> ScipyDist:
    """Create a normal distribution.

    Args:
        loc: Mean of the distribution
        scale: Standard deviation

    Returns:
        ScipyDist wrapper for scipy.stats.norm
    """
    return ScipyDist(scipy_stats.norm, loc=loc, scale=scale)


def Uniform(loc: float = 0.0, scale: float = 1.0) -> ScipyDist:
    """Create a uniform distribution.

    Args:
        loc: Lower bound
        scale: Width of the distribution (upper = loc + scale)

    Returns:
        ScipyDist wrapper for scipy.stats.uniform
    """
    return ScipyDist(scipy_stats.uniform, loc=loc, scale=scale)


def Exponential(scale: float = 1.0) -> ScipyDist:
    """Create an exponential distribution.

    Args:
        scale: Scale parameter (1/rate)

    Returns:
        ScipyDist wrapper for scipy.stats.expon
    """
    return ScipyDist(scipy_stats.expon, scale=scale)


def Poisson(mu: float) -> ScipyDist:
    """Create a Poisson distribution.

    Args:
        mu: Mean/rate parameter

    Returns:
        ScipyDist wrapper for scipy.stats.poisson
    """
    return ScipyDist(scipy_stats.poisson, mu=mu)
