"""Integration module for zfit2.

This module provides functions for integrating models and functions,
using both analytical and numerical methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from zfit2.backend import numpy as znp
from zfit2.func import Func
from zfit2.parameter import Parameter

from .numerical import (
    integrate_gauss_legendre,
    integrate_simpson,
    integrate_trapezoid,
    integrate_monte_carlo,
    integrate_vegas
)


def integrate(
        func: Func,
        limits: Union[Tuple[float, float], List[Tuple[float, float]]],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        method: str = "auto",
        **kwargs
) -> float:
    """Integrate a function over the specified limits.

    Args:
        func: Function to integrate
        limits: Integration limits
        parameters: Parameters for the function
        method: Integration method ('auto', 'analytic', 'numerical',
                                   'gauss_legendre', 'simpson', 'trapezoid',
                                   'monte_carlo', 'vegas')
        **kwargs: Additional arguments for the integration method

    Returns:
        The integral value
    """
    # Try analytic integration if available and requested
    if method in ("auto", "analytic") and hasattr(func, "integral") and func.properties.has_analytic_integral:
        try:
            result = func.integral(limits, parameters=parameters, **kwargs)
            return result
        except (NotImplementedError, ValueError, TypeError) as e:
            if method == "analytic":
                raise ValueError(f"Analytic integration failed: {e}")
            # Fall back to numerical integration

    # Use numerical integration
    return integrate_numerically(
        func, limits, parameters=parameters,
        method=(None if method == "auto" else method),
        **kwargs
    )


def integrate_numerically(
        func: Func,
        limits: Union[Tuple[float, float], List[Tuple[float, float]]],
        *,
        parameters: Optional[Dict[str, Any]] = None,
        method: Optional[str] = None,
        **kwargs
) -> float:
    """Integrate a function numerically over the specified limits.

    Args:
        func: Function to integrate
        limits: Integration limits
        parameters: Parameters for the function
        method: Integration method (None = auto-select, 'gauss_legendre', 'simpson',
                                   'trapezoid', 'monte_carlo', 'vegas')
        **kwargs: Additional arguments for the integration method

    Returns:
        The integral value
    """
    # Ensure limits is a list of tuples
    if isinstance(limits, tuple) and len(limits) == 2 and all(isinstance(x, (int, float)) for x in limits):
        limits = [limits]

    # Determine dimensionality
    ndim = len(limits)

    # Create a wrapper function for the integrand
    def integrand(*x_values):
        x_dict = {}
        for i, var in enumerate(func.domain.variables):
            x_dict[var.name] = x_values[i]
        return func(x_dict, parameters=parameters)

    # Select appropriate method based on dimensionality
    if method is None:
        if ndim == 1:
            method = "gauss_legendre"
        else:
            method = "vegas"

    # Call the appropriate integration function
    if method == "gauss_legendre" and ndim == 1:
        return integrate_gauss_legendre(integrand, limits[0], **kwargs)
    elif method == "simpson" and ndim == 1:
        return integrate_simpson(integrand, limits[0], **kwargs)
    elif method == "trapezoid" and ndim == 1:
        return integrate_trapezoid(integrand, limits[0], **kwargs)
    elif method == "monte_carlo":
        return integrate_monte_carlo(integrand, limits, **kwargs)
    elif method == "vegas":
        return integrate_vegas(integrand, limits, **kwargs)
    else:
        raise ValueError(f"Unsupported integration method '{method}' for {ndim}-dimensional integral")