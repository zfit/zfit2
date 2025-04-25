"""
Utility functions for jax-hist.

This module provides utility functions for working with histograms,
including JAX transformations and convenience functions.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from . import axis, histogram


def hist(
    data: Union[jnp.ndarray, np.ndarray, Sequence[float]],
    bins: Union[int, List[float], List[List[float]]],
    range: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    weights: Optional[Union[jnp.ndarray, np.ndarray, Sequence[float]]] = None,
) -> histogram.Hist:
    """Create a histogram from data like numpy.histogram.
    
    Args:
        data: Data to histogram
        bins: Number of bins or bin edges
        range: Range of values to histogram
        weights: Weights for each data point
    
    Returns:
        A new histogram
    """
    # Convert to arrays
    data_array = jnp.asarray(data)
    
    # Handle 1D data
    if data_array.ndim == 1:
        # Create axis
        if isinstance(bins, int):
            # Regular axis with specified number of bins
            if range is None:
                start, stop = jnp.min(data_array), jnp.max(data_array)
            else:
                start, stop = range
            ax = axis.RegularAxis(bins, start, stop)
        else:
            # Variable axis with specified bin edges
            ax = axis.VariableAxis(bins)
        
        # Create histogram
        hist = histogram.Hist(ax)
        
        # Fill histogram
        if weights is None:
            hist.fill(data_array)
        else:
            weights_array = jnp.asarray(weights)
            hist.fill(data_array, weight=weights_array)
        
        return hist
    
    # Handle multi-dimensional data
    else:
        # Check that data has the right shape
        n_dims = data_array.shape[1]
        
        # Create axes
        axes = []
        for dim in range(n_dims):
            dim_data = data_array[:, dim]
            
            if isinstance(bins, int):
                # Same number of bins for all dimensions
                if range is None:
                    start, stop = jnp.min(dim_data), jnp.max(dim_data)
                else:
                    start, stop = range[dim]
                axes.append(axis.RegularAxis(bins, start, stop))
            elif isinstance(bins[0], (int, float)):
                # Same bin edges for all dimensions
                axes.append(axis.VariableAxis(bins))
            else:
                # Different bin edges for each dimension
                axes.append(axis.VariableAxis(bins[dim]))
        
        # Create histogram
        hist = histogram.Hist(*axes)
        
        # Fill histogram - need to transpose data to get it in the right shape
        if weights is None:
            for i in range(len(data_array)):
                hist.fill(*data_array[i])
        else:
            weights_array = jnp.asarray(weights)
            for i in range(len(data_array)):
                hist.fill(*data_array[i], weight=weights_array[i])
        
        return hist


def histogram2d(
    x: Union[jnp.ndarray, np.ndarray, Sequence[float]],
    y: Union[jnp.ndarray, np.ndarray, Sequence[float]],
    bins: Union[int, Tuple[int, int], List[float], Tuple[List[float], List[float]]],
    range: Optional[Union[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    weights: Optional[Union[jnp.ndarray, np.ndarray, Sequence[float]]] = None,
) -> histogram.Hist:
    """Create a 2D histogram like numpy.histogram2d.
    
    Args:
        x: x coordinates
        y: y coordinates
        bins: Number of bins or bin edges for each dimension
        range: Range of values to histogram for each dimension
        weights: Weights for each data point
    
    Returns:
        A new 2D histogram
    """
    # Convert to arrays
    x_array = jnp.asarray(x)
    y_array = jnp.asarray(y)
    
    # Stack data
    data = jnp.stack([x_array, y_array], axis=1)
    
    # Create histogram
    return hist(data, bins, range, weights)


def histogramdd(
    sample: Union[jnp.ndarray, np.ndarray, Sequence[Sequence[float]]],
    bins: Union[int, Sequence[int], Sequence[Sequence[float]]],
    range: Optional[Sequence[Tuple[float, float]]] = None,
    weights: Optional[Union[jnp.ndarray, np.ndarray, Sequence[float]]] = None,
) -> histogram.Hist:
    """Create a multi-dimensional histogram like numpy.histogramdd.
    
    Args:
        sample: Data points
        bins: Number of bins or bin edges for each dimension
        range: Range of values to histogram for each dimension
        weights: Weights for each data point
    
    Returns:
        A new multi-dimensional histogram
    """
    # Convert to array
    sample_array = jnp.asarray(sample)
    
    # Create histogram
    return hist(sample_array, bins, range, weights)


def jit_fill(hist_func: Callable[..., histogram.Hist]) -> Callable:
    """Decorator to create a jit-compiled histogram filling function.
    
    This decorator takes a function that returns a histogram and
    replaces it with a function that returns a jit-compiled
    version of the histogram filling code.
    
    Args:
        hist_func: A function that returns a histogram
    
    Returns:
        A decorated function that returns a jit-compiled histogram
    """
    @functools.wraps(hist_func)
    def wrapper(*args, **kwargs):
        # Create the histogram
        hist = hist_func(*args, **kwargs)
        
        # Define a function to fill the histogram
        def fill_func(data, weights=None):
            nonlocal hist
            if weights is None:
                hist.fill(data)
            else:
                hist.fill(data, weight=weights)
            return hist
        
        # Compile the fill function
        jitted_fill = jax.jit(fill_func)
        
        # Replace the histogram's fill method
        hist.jitted_fill = jitted_fill
        
        return hist
    
    return wrapper


# Register histogram types as JAX PyTrees
def _hist_flatten(hist: histogram.Hist):
    """Flatten a histogram for JAX PyTree."""
    children = (hist._counts,)
    aux_data = {
        "axes": hist.axes,
        "name": hist.name,
        "label": hist.label,
        "metadata": hist.metadata,
    }
    return children, aux_data


def _hist_unflatten(aux_data, children):
    """Unflatten a histogram from JAX PyTree."""
    counts, = children
    hist = histogram.Hist(
        *aux_data["axes"],
        data=counts,
        name=aux_data["name"],
        label=aux_data["label"],
        metadata=aux_data["metadata"],
    )
    return hist


# Register the PyTree
try:
    jax.tree_util.register_pytree_node(
        histogram.Hist,
        _hist_flatten,
        _hist_unflatten
    )
except:
    # Registration already exists or failed
    pass


def grad_bin_count(hist: histogram.Hist, value: Union[float, jnp.ndarray]) -> Callable:
    """Create a function that computes the gradient of a bin's count.
    
    Args:
        hist: Histogram
        value: Value to get the bin for
    
    Returns:
        A function that takes a parameter and returns the gradient
    """
    def bin_count(param):
        """Compute the bin count for a given parameter."""
        # Compute the bin index
        idx = hist.axes[0].index(value * param)
        
        # Get the bin count
        return hist._counts[idx]
    
    return jax.grad(bin_count)


def vmap_fill(hist: histogram.Hist) -> Callable:
    """Create a vectorized fill function for a histogram.
    
    Args:
        hist: Histogram to fill
    
    Returns:
        A vectorized fill function
    """
    # Define a fill function for a single data point
    def fill_one(data_point, weight=None):
        """Fill the histogram with a single data point."""
        if weight is None:
            hist.fill(*data_point)
        else:
            hist.fill(*data_point, weight=weight)
        return hist
    
    # Vectorize the fill function
    return jax.vmap(fill_one)
