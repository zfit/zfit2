"""
JAX-friendly histogram implementation that is fully compatible with JAX transformations.
"""

from __future__ import annotations

import functools
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np


class Axis:
    """Base class for histogram axes."""

    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        underflow: bool = True,
        overflow: bool = True,
    ):
        self.bins = bins
        self.start = start
        self.stop = stop
        self.name = name
        self.label = label if label else name
        self.underflow = underflow
        self.overflow = overflow

    @property
    def n_bins(self) -> int:
        """Get the number of bins (excluding flow bins)."""
        return self.bins

    @property
    def extent(self) -> int:
        """Get the total number of bins (including flow bins)."""
        return self.bins + self.underflow + self.overflow

    @property
    def edges(self) -> jnp.ndarray:
        """Get the bin edges."""
        return jnp.linspace(self.start, self.stop, self.bins + 1)

    @property
    def centers(self) -> jnp.ndarray:
        """Get the bin centers."""
        edges = self.edges
        return (edges[:-1] + edges[1:]) / 2

    @property
    def widths(self) -> jnp.ndarray:
        """Get the bin widths."""
        edges = self.edges
        return edges[1:] - edges[:-1]

    def index(self, value: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """Get the bin index for a value."""
        value = jnp.asarray(value, dtype=jnp.float32)

        # Calculate the bin index
        bin_width = (self.stop - self.start) / self.bins
        raw_index = (value - self.start) / bin_width

        # Ensure int32 type for indexing
        index = jnp.floor(raw_index).astype(jnp.int32)

        # Handle underflow/overflow
        index = jnp.where(
            jnp.logical_and(index >= 0, index < self.bins),
            index,
            jnp.where(
                index < 0,
                # Underflow handling
                jnp.where(
                    self.underflow,
                    jnp.array(-1, dtype=jnp.int32),  # Underflow bin
                    jnp.zeros_like(index, dtype=jnp.int32),  # First bin
                ),
                # Overflow handling
                jnp.where(
                    self.overflow,
                    jnp.array(self.bins, dtype=jnp.int32),  # Overflow bin
                    jnp.array(self.bins - 1, dtype=jnp.int32),  # Last bin
                ),
            ),
        )

        return index


class JaxHist:
    """JAX-friendly histogram that works with JAX transformations."""

    def __init__(
        self,
        *axes: Axis,
        counts: Optional[jnp.ndarray] = None,
        name: str = "",
        label: str = "",
    ):
        self.axes = list(axes)
        self.name = name
        self.label = label if label else name

        # Initialize storage
        shape = tuple(ax.extent for ax in self.axes)
        if counts is not None:
            if counts.shape != shape:
                raise ValueError(
                    f"Counts shape {counts.shape} doesn't match expected shape {shape}"
                )
            self._counts = counts
        else:
            self._counts = jnp.zeros(shape, dtype=jnp.float32)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self.axes)

    @property
    def values(self) -> jnp.ndarray:
        """Get the bin values."""
        return self._counts

    @property
    def variances(self) -> jnp.ndarray:
        """Get the bin variances (same as values for simple counts)."""
        return self._counts

    def fill(
        self, values: list[jnp.ndarray], weights: Optional[jnp.ndarray] = None
    ) -> JaxHist:
        """Fill the histogram using a pure JAX approach.

        Args:
            values: List of value arrays, one for each axis
            weights: Optional weights

        Returns:
            Updated histogram
        """
        if len(values) != self.ndim:
            raise ValueError(f"Expected {self.ndim} value arrays, got {len(values)}")

        # Convert to arrays
        arrays = [jnp.asarray(val, dtype=jnp.float32) for val in values]

        # Find common length
        shapes = [arr.shape[0] for arr in arrays]
        max_len = max(shapes)

        # Prepare weights
        if weights is None:
            weights_array = jnp.ones(max_len, dtype=jnp.float32)
        else:
            weights_array = jnp.asarray(weights, dtype=jnp.float32)
            if weights_array.shape[0] == 1:
                weights_array = jnp.broadcast_to(weights_array, (max_len,))

        # Create a pure function to add a single point
        def add_point(counts, i):
            indices = tuple(
                ax.index(arr[i % arr.shape[0]])
                for ax, arr in zip(self.axes, arrays, strict=False)
            )
            weight = weights_array[i % weights_array.shape[0]]
            return counts.at[indices].add(weight)

        # Use jax.lax.fori_loop for a JAX-friendly loop
        new_counts = jax.lax.fori_loop(
            0, max_len, lambda i, counts: add_point(counts, i), self._counts
        )

        # Create a new histogram with updated counts
        return JaxHist(*self.axes, counts=new_counts, name=self.name, label=self.label)

    def fill_np(
        self, values: list[np.ndarray], weights: Optional[np.ndarray] = None
    ) -> JaxHist:
        """Fill the histogram using numpy for intermediate calculations.

        Args:
            values: List of value arrays, one for each axis
            weights: Optional weights

        Returns:
            Updated histogram
        """
        if len(values) != self.ndim:
            raise ValueError(f"Expected {self.ndim} value arrays, got {len(values)}")

        # Use numpy for intermediate calculations
        counts = np.array(self._counts)

        # Convert to numpy arrays
        arrays = [np.asarray(val, dtype=np.float32) for val in values]

        # Find common length
        shapes = [arr.shape[0] for arr in arrays]
        max_len = max(shapes)

        # Prepare weights
        if weights is None:
            weights_array = np.ones(max_len, dtype=np.float32)
        else:
            weights_array = np.asarray(weights, dtype=np.float32)
            if weights_array.shape[0] == 1:
                weights_array = np.broadcast_to(weights_array, (max_len,))

        # Fill the histogram
        for i in range(max_len):
            try:
                indices = tuple(
                    int(ax.index(arr[i % arr.shape[0]]))
                    for ax, arr in zip(self.axes, arrays, strict=False)
                )
                weight = float(weights_array[i % weights_array.shape[0]])
                counts[indices] += weight
            except (IndexError, ValueError, TypeError):
                continue

        # Create a new histogram with updated counts
        return JaxHist(
            *self.axes, counts=jnp.array(counts), name=self.name, label=self.label
        )

    def project(self, axis_indices: list[int]) -> JaxHist:
        """Project the histogram onto the specified axes.

        Args:
            axis_indices: Indices of axes to keep

        Returns:
            Projected histogram
        """
        # Validate indices
        if not all(0 <= idx < self.ndim for idx in axis_indices):
            raise ValueError(f"Axis indices must be between 0 and {self.ndim - 1}")

        # Get axes to keep
        new_axes = [self.axes[i] for i in axis_indices]

        # Get axes to sum over
        sum_axes = tuple(i for i in range(self.ndim) if i not in axis_indices)

        # Project by summing over other axes
        new_counts = jnp.sum(self._counts, axis=sum_axes)

        # Create new histogram
        return JaxHist(*new_axes, counts=new_counts, name=self.name, label=self.label)

    def __add__(self, other: Union[JaxHist, float]) -> JaxHist:
        """Add two histograms or add a scalar."""
        if isinstance(other, JaxHist):
            if len(self.axes) != len(other.axes):
                raise ValueError("Cannot add histograms with different dimensions")

            new_counts = self._counts + other._counts
        else:
            new_counts = self._counts + other

        return JaxHist(*self.axes, counts=new_counts, name=self.name, label=self.label)

    def __mul__(self, other: Union[JaxHist, float]) -> JaxHist:
        """Multiply two histograms or multiply by a scalar."""
        if isinstance(other, JaxHist):
            if len(self.axes) != len(other.axes):
                raise ValueError("Cannot multiply histograms with different dimensions")

            new_counts = self._counts * other._counts
        else:
            new_counts = self._counts * other

        return JaxHist(*self.axes, counts=new_counts, name=self.name, label=self.label)

    def __truediv__(self, other: Union[JaxHist, float]) -> JaxHist:
        """Divide two histograms or divide by a scalar."""
        if isinstance(other, JaxHist):
            if len(self.axes) != len(other.axes):
                raise ValueError("Cannot divide histograms with different dimensions")

            # Handle division by zero using jnp.where (JAX does not support 'out' or 'where' in jnp.divide)
            new_counts = jnp.where(
                other._counts != 0, self._counts / other._counts, 0.0
            )
        else:
            new_counts = self._counts / other

        return JaxHist(*self.axes, counts=new_counts, name=self.name, label=self.label)


# Register JaxHist as a JAX PyTree
def _hist_flatten(hist):
    """Flatten a JaxHist for JAX PyTree."""
    children = (hist._counts,)
    aux_data = {
        "axes": hist.axes,
        "name": hist.name,
        "label": hist.label,
    }
    return children, aux_data


def _hist_unflatten(aux_data, children):
    """Unflatten a JaxHist from JAX PyTree."""
    (counts,) = children
    return JaxHist(
        *aux_data["axes"], counts=counts, name=aux_data["name"], label=aux_data["label"]
    )


jax.tree_util.register_pytree_node(JaxHist, _hist_flatten, _hist_unflatten)


# Convenience functions
def hist(
    x: jnp.ndarray,
    bins: int = 10,
    range: Optional[tuple[float, float]] = None,
    weights: Optional[jnp.ndarray] = None,
) -> JaxHist:
    """Create a 1D histogram."""
    x_arr = jnp.asarray(x)

    if range is None:
        start, stop = jnp.min(x_arr), jnp.max(x_arr)
    else:
        start, stop = range

    axis = Axis(bins, start, stop)
    h = JaxHist(axis)

    return h.fill([x_arr], weights)


def hist2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    bins: Union[int, tuple[int, int]] = 10,
    range: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    weights: Optional[jnp.ndarray] = None,
) -> JaxHist:
    """Create a 2D histogram."""
    x_arr = jnp.asarray(x)
    y_arr = jnp.asarray(y)

    # Handle bins
    if isinstance(bins, int):
        bins_x = bins_y = bins
    else:
        bins_x, bins_y = bins

    # Handle range
    if range is None:
        x_range = (jnp.min(x_arr), jnp.max(x_arr))
        y_range = (jnp.min(y_arr), jnp.max(y_arr))
    else:
        x_range, y_range = range

    x_axis = Axis(bins_x, x_range[0], x_range[1], name="x")
    y_axis = Axis(bins_y, y_range[0], y_range[1], name="y")

    h = JaxHist(x_axis, y_axis)

    return h.fill([x_arr, y_arr], weights)


# JAX transformation helpers
def jit_histogram(func):
    """Decorator to jit-compile a function that returns a histogram."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return jax.jit(func)(*args, **kwargs)

    return wrapper


def vmap_histogram(func, in_axes=0, out_axes=0):
    """Vectorize a function that processes histograms."""
    return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def grad_histogram(func, argnums=0):
    """Get the gradient of a function that returns a histogram."""
    return jax.grad(func, argnums=argnums)
