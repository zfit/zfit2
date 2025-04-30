"""Sampling module for zfit2."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from zfit2.dist import Distribution


def rejection_sample(
    dist: Distribution,
    size: int,
    *,
    params: Optional[dict[str, Any]] = None,
    norm: Optional[list[tuple[float, float]]] = None,
    max_iterations: int = 10,
    seed: int = 42,
) -> dict[str, jnp.ndarray]:
    """Sample from a distribution using rejection sampling.

    Args:
        dist: Distribution to sample from
        size: Number of samples to generate
        params: Parameters for the distribution
        norm: Normalization range
        max_iterations: Maximum number of iterations
        seed: Random seed

    Returns:
        Dictionary of samples for each variable
    """
    # Get domain variables
    variables = dist.domain.variables
    var_names = [var.name for var in variables]

    # Get sampling limits
    limits = []
    for var in variables:
        if var.lower is None or var.upper is None:
            raise ValueError(
                f"Variable {var.name} must have finite bounds for rejection sampling"
            )
        limits.append((var.lower, var.upper))

    # Find maximum PDF value
    # For now, we use a simple grid search
    n_grid = min(100, max(10, int(size ** (1 / len(variables)))))
    grid_points = []
    for lower, upper in limits:
        grid_points.append(jnp.linspace(lower, upper, n_grid))

    max_pdf = 0.0
    for indices in np.ndindex(*[len(points) for points in grid_points]):
        point = {var_names[i]: grid_points[i][idx] for i, idx in enumerate(indices)}
        pdf_val = dist.pdf(point, params=params, norm=norm)
        max_pdf = max(max_pdf, float(pdf_val))

    # Add a safety factor
    max_pdf *= 1.2

    # Initialize random key
    key = jax.random.key(seed)

    # Perform rejection sampling
    accepted_samples = {name: [] for name in var_names}
    n_accepted = 0

    for _ in range(max_iterations):
        # Determine how many samples to generate in this iteration
        n_remaining = size - n_accepted
        if n_remaining <= 0:
            break

        # Generate uniform random points in the domain
        points = {}
        for i, (name, (lower, upper)) in enumerate(
            zip(var_names, limits, strict=False)
        ):
            key, subkey = jax.random.split(key)
            points[name] = jax.random.uniform(
                subkey, shape=(n_remaining * 10,), minval=lower, maxval=upper
            )

        # Evaluate PDF at random points
        pdf_values = dist.pdf(points, params=params, norm=norm)

        # Generate uniform random values for rejection test
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(
            subkey, shape=(n_remaining * 10,), minval=0.0, maxval=max_pdf
        )

        # Accept points where u <= pdf
        accept_mask = u <= pdf_values

        for name in var_names:
            accepted_samples[name].extend(points[name][accept_mask][:n_remaining])

        # Update number of accepted samples
        n_accepted += int(jnp.sum(accept_mask))
        if n_accepted >= size:
            # Truncate to exact size
            for name in var_names:
                accepted_samples[name] = accepted_samples[name][:size]
            break

    # Convert lists to arrays
    result = {name: jnp.array(samples) for name, samples in accepted_samples.items()}

    return result
