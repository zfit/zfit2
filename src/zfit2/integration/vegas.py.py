"""Vegas algorithm for multi-dimensional numerical integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from zfit2.backend import numpy as znp


class VegasIntegrator:
    """Vegas algorithm for multi-dimensional integration.
    
    The Vegas algorithm uses importance sampling to reduce the variance
    of the Monte Carlo estimate. It adapts the sampling grid to
    concentrate points in regions where the integrand is large.
    """
    
    def __init__(
        self,
        func: Callable,
        limits: List[Tuple[float, float]],
        n_bins: int = 50,
        alpha: float = 1.5,
        beta: float = 0.75,
        seed: int = 42
    ):
        """Initialize Vegas integrator.
        
        Args:
            func: Function to integrate
            limits: Integration limits for each dimension
            n_bins: Number of bins per dimension
            alpha: Grid adaptation parameter (1.5 recommended)
            beta: Learning rate parameter (0.75 recommended)
            seed: Random seed
        """
        self.func = func
        self.limits = limits
        self.ndim = len(limits)
        self.n_bins = n_bins
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        
        # Calculate volume of integration region
        self.volume = 1.0
        for lower, upper in limits:
            self.volume *= (upper - lower)
        
        # Initialize random state
        self.key = jax.random.key(seed)
        
        # Initialize grid
        self._init_grid()
        
        # Storage for results
        self.results = []
        self.errors = []
        self.chi_sq = []
        
        # JIT-compile functions
        self._jit_sample_and_evaluate = jax.jit(self._sample_and_evaluate)
        self._jit_update_grid = jax.jit(self._update_grid)
    
    def _init_grid(self):
        """Initialize the integration grid."""
        # Each dimension has n_bins bins with uniform probability 1/n_bins
        self.grid = []
        for d in range(self.ndim):
            lower, upper = self.limits[d]
            # Create uniform grid
            edges = jnp.linspace(lower, upper, self.n_bins + 1)
            # Initialize grid with uniform probability density
            self.grid.append(jnp.ones(self.n_bins) / self.n_bins)
    
    def _transform_grid(self):
        """Transform grid probabilities to bin edges."""
        edges = []
        for d in range(self.ndim):
            lower, upper = self.limits[d]
            # Calculate cumulative probabilities
            cum_probs = jnp.cumsum(self.grid[d])
            cum_probs = jnp.concatenate([jnp.array([0.0]), cum_probs])
            # Transform to bin edges
            dim_edges = lower + cum_probs * (upper - lower)
            edges.append(dim_edges)
        return edges
    
    def _sample_and_evaluate(
        self, 
        key: jnp.ndarray, 
        n_samples: int, 
        edges: List[jnp.ndarray],
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample points and evaluate function with importance sampling."""
        # Generate random bin indices and positions within bins
        bin_indices = []
        bin_positions = []
        points = []
        jacobian_factors = []
        
        for d in range(self.ndim):
            # Split key for this dimension
            key, subkey = jax.random.split(key)
            
            # Sample bin indices
            dim_indices = jax.random.randint(
                subkey, shape=(n_samples,), minval=0, maxval=self.n_bins
            )
            
            # Split key for sampling within bins
            key, subkey = jax.random.split(key)
            
            # Sample positions within bins (uniform from 0 to 1)
            dim_positions = jax.random.uniform(
                subkey, shape=(n_samples,), minval=0.0, maxval=1.0
            )
            
            # Calculate actual coordinates
            dim_edges = edges[d]
            lower_edges = dim_edges[dim_indices]
            upper_edges = dim_edges[dim_indices + 1]
            dim_points = lower_edges + dim_positions * (upper_edges - lower_edges)
            
            # Calculate Jacobian factor for this dimension
            # (correction for non-uniform sampling)
            bin_widths = upper_edges - lower_edges
            dim_jacobian = self.n_bins / bin_widths
            
            # Store results
            bin_indices.append(dim_indices)
            bin_positions.append(dim_positions)
            points.append(dim_points)
            jacobian_factors.append(dim_jacobian)
        
        # Calculate full Jacobian and weighted values
        jacobian = jnp.ones(n_samples)
        for d in range(self.ndim):
            jacobian *= jacobian_factors[d]
        
        # Evaluate function at the sampled points
        func_values = jnp.array([
            self.func(*[points[d][i] for d in range(self.ndim)], **kwargs)
            for i in range(n_samples)
        ])
        
        # Weight function values by Jacobian
        weighted_values = func_values * jacobian
        
        return (
            jnp.array(bin_indices), 
            jnp.array(weighted_values), 
            jnp.array(func_values)
        )
    
    def _update_grid(
        self, 
        bin_indices: jnp.ndarray, 
        weighted_values: jnp.ndarray, 
        func_values: jnp.ndarray
    ):
        """Update grid using importance sampling."""
        n_samples = bin_indices.shape[1]
        
        # For each dimension, update the grid
        new_grid = []
        for d in range(self.ndim):
            dim_indices = bin_indices[d]
            
            # Create histogram of function values weighted by Jacobian
            bins = jnp.zeros(self.n_bins)
            
            for i in range(n_samples):
                bin_idx = dim_indices[i]
                bins = bins.at[bin_idx].add(weighted_values[i])
            
            # Normalize bins
            bins = bins / jnp.sum(bins)
            
            # Compute variance reduction factors
            # We want to sample more where the function varies more
            if jnp.sum(bins) > 0:
                # Avoid division by zero
                bins = jnp.where(bins > 0, bins, 1e-10)
                
                # Calculate the weighted variance reduction factor
                f_avg = jnp.mean(weighted_values)
                variance_factor = jnp.zeros(self.n_bins)
                
                for i in range(n_samples):
                    bin_idx = dim_indices[i]
                    variance_factor = variance_factor.at[bin_idx].add(
                        (weighted_values[i] - f_avg) ** 2
                    )
                
                variance_factor = jnp.sqrt(variance_factor / n_samples)
                
                # Apply adaptation algorithm
                grid_delta = ((variance_factor / bins) ** self.alpha) * bins
                
                # Normalize
                grid_delta = grid_delta / jnp.sum(grid_delta)
                
                # Update grid using learning rate
                updated_grid = self.grid[d] * (1 - self.beta) + grid_delta * self.beta
                
                # Normalize again
                updated_grid = updated_grid / jnp.sum(updated_grid)
            else:
                # If all bins are zero, keep uniform grid
                updated_grid = jnp.ones(self.n_bins) / self.n_bins
            
            new_grid.append(updated_grid)
        
        self.grid = new_grid
    
    def integrate(
        self, 
        n_samples: int = 10000, 
        n_iterations: int = 10,
        **kwargs
    ) -> Tuple[float, float]:
        """Perform Vegas integration.
        
        Args:
            n_samples: Number of samples per iteration
            n_iterations: Number of iterations
            **kwargs: Additional arguments for the function
        
        Returns:
            Tuple of (integral estimate, error estimate)
        """
        # Reset results
        self.results = []
        self.errors = []
        self.chi_sq = []
        
        # Perform integration iterations
        for iter in range(n_iterations):
            # Generate new random key
            self.key, subkey = jax.random.split(self.key)
            
            # Transform grid to edges
            edges = self._transform_grid()
            
            # Sample points and evaluate function
            bin_indices, weighted_values, func_values = self._jit_sample_and_evaluate(
                subkey, n_samples, edges, **kwargs
            )
            
            # Calculate integral estimate for this iteration
            iter_result = self.volume * jnp.mean(weighted_values)
            iter_error = self.volume * jnp.std(weighted_values) / jnp.sqrt(n_samples)
            
            # Store results
            self.results.append(float(iter_result))
            self.errors.append(float(iter_error))
            
            # Update grid
            if iter < n_iterations - 1:  # Don't update after last iteration
                self._jit_update_grid(bin_indices, weighted_values, func_values)
        
        # Combine results from iterations (weighted by inverse variance)
        weights = jnp.array([1.0 / (err**2) for err in self.errors])
        weighted_sum = jnp.sum(jnp.array(self.results) * weights)
        weight_sum = jnp.sum(weights)
        
        # Final estimate and error
        final_result = weighted_sum / weight_sum
        final_error = jnp.sqrt(1.0 / weight_sum)
        
        return float(final_result), float(final_error)


def integrate_vegas(
    func: Callable,
    limits: List[Tuple[float, float]],
    *,
    n_samples: int = 10000,
    n_iterations: int = 10,
    n_bins: int = 50,
    alpha: float = 1.5,
    beta: float = 0.75,
    seed: int = 42,
    **kwargs
) -> float:
    """Integrate a function using the Vegas algorithm.
    
    Args:
        func: Function to integrate
        limits: Integration limits for each dimension
        n_samples: Number of samples per iteration
        n_iterations: Number of iterations
        n_bins: Number of bins per dimension
        alpha: Grid adaptation parameter (1.5 recommended)
        beta: Learning rate parameter (0.75 recommended)
        seed: Random seed
        **kwargs: Additional arguments for the function
    
    Returns:
        The integral value
    """
    # Create Vegas integrator
    integrator = VegasIntegrator(
        func=func,
        limits=limits,
        n_bins=n_bins,
        alpha=alpha,
        beta=beta,
        seed=seed
    )
    
    # Perform integration
    result, error = integrator.integrate(
        n_samples=n_samples,
        n_iterations=n_iterations,
        **kwargs
    )
    
    return result