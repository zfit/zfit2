"""Numerical integration methods for zfit2."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


def integrate_gauss_legendre(
    func: Callable, limits: tuple[float, float], *, n_points: int = 100, **kwargs
) -> float:
    """Integrate a function using Gauss-Legendre quadrature.

    Args:
        func: Function to integrate
        limits: Integration limits (lower, upper)
        n_points: Number of quadrature points
        **kwargs: Additional arguments for the function

    Returns:
        The integral value
    """
    # Get Gauss-Legendre quadrature points and weights
    points, weights = np.polynomial.legendre.leggauss(n_points)

    # Map points from [-1, 1] to [lower, upper]
    lower, upper = limits
    scaled_points = 0.5 * (upper - lower) * points + 0.5 * (upper + lower)
    scaled_weights = 0.5 * (upper - lower) * weights

    # Evaluate function at quadrature points
    func_values = jnp.array([func(x, **kwargs) for x in scaled_points])

    # Compute weighted sum
    return jnp.sum(func_values * scaled_weights)


def integrate_simpson(
    func: Callable, limits: tuple[float, float], *, n_points: int = 101, **kwargs
) -> float:
    """Integrate a function using Simpson's rule.

    Args:
        func: Function to integrate
        limits: Integration limits (lower, upper)
        n_points: Number of points (must be odd)
        **kwargs: Additional arguments for the function

    Returns:
        The integral value
    """
    # Ensure odd number of points
    if n_points % 2 == 0:
        n_points += 1

    # Create grid of evaluation points
    lower, upper = limits
    grid = jnp.linspace(lower, upper, n_points)

    # Evaluate function at grid points
    func_values = jnp.array([func(x, **kwargs) for x in grid])

    # Apply Simpson's rule
    h = (upper - lower) / (n_points - 1)

    # Weights: 1, 4, 2, 4, 2, ..., 4, 1
    weights = jnp.ones(n_points)
    weights = weights.at[1::2].set(4.0)
    weights = weights.at[2::2].set(2.0)
    weights = weights.at[-1].set(1.0)

    return h / 3.0 * jnp.sum(weights * func_values)


def integrate_trapezoid(
    func: Callable, limits: tuple[float, float], *, n_points: int = 100, **kwargs
) -> float:
    """Integrate a function using the trapezoidal rule.

    Args:
        func: Function to integrate
        limits: Integration limits (lower, upper)
        n_points: Number of points
        **kwargs: Additional arguments for the function

    Returns:
        The integral value
    """
    # Create grid of evaluation points
    lower, upper = limits
    grid = jnp.linspace(lower, upper, n_points)

    # Evaluate function at grid points
    func_values = jnp.array([func(x, **kwargs) for x in grid])

    # Apply trapezoidal rule
    h = (upper - lower) / (n_points - 1)
    return h * (
        0.5 * func_values[0] + jnp.sum(func_values[1:-1]) + 0.5 * func_values[-1]
    )


def integrate_monte_carlo(
    func: Callable,
    limits: list[tuple[float, float]],
    *,
    n_points: int = 10000,
    seed: int = 42,
    **kwargs,
) -> float:
    """Integrate a function using Monte Carlo integration.

    Args:
        func: Function to integrate
        limits: Integration limits for each dimension
        n_points: Number of random sampling points
        seed: Random seed
        **kwargs: Additional arguments for the function

    Returns:
        The integral value
    """
    # Get dimensionality
    ndim = len(limits)

    # Calculate volume of integration region
    volume = 1.0
    for lower, upper in limits:
        volume *= upper - lower

    # Generate random points in the hypercube
    key = jax.random.key(seed)
    points = []

    for i, (lower, upper) in enumerate(limits):
        key, subkey = jax.random.split(key)
        points.append(
            jax.random.uniform(subkey, shape=(n_points,), minval=lower, maxval=upper)
        )

    # Evaluate function at random points
    func_values = jnp.array(
        [func(*[points[d][i] for d in range(ndim)], **kwargs) for i in range(n_points)]
    )

    # Compute Monte Carlo estimate
    integral = volume * jnp.mean(func_values)

    return integral
