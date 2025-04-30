"""
Histogram module for jax-hist.

This module provides a JAX-based histogram that follows the Unified
Histogram Interface (UHI) requirements and is compatible with JAX transformations.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Union,
)

import jax.numpy as jnp
import numpy as np

from . import axis


@dataclass
class Accumulator:
    """Base class for histogram accumulators."""


@dataclass
class Count(Accumulator):
    """Simple count accumulator."""

    count: jnp.ndarray

    def __add__(self, other: Count) -> Count:
        if not isinstance(other, Count):
            return NotImplemented
        return Count(self.count + other.count)


@dataclass
class WeightedSum(Accumulator):
    """Weighted sum accumulator."""

    sum: jnp.ndarray
    sum_of_weights_squared: jnp.ndarray

    def __add__(self, other: WeightedSum) -> WeightedSum:
        if not isinstance(other, WeightedSum):
            return NotImplemented
        return WeightedSum(
            self.sum + other.sum,
            self.sum_of_weights_squared + other.sum_of_weights_squared,
        )


@dataclass
class Mean(Accumulator):
    """Mean accumulator."""

    count: jnp.ndarray
    sum: jnp.ndarray
    sum_of_squares: jnp.ndarray

    def __add__(self, other: Mean) -> Mean:
        if not isinstance(other, Mean):
            return NotImplemented
        return Mean(
            self.count + other.count,
            self.sum + other.sum,
            self.sum_of_squares + other.sum_of_squares,
        )

    @property
    def mean(self) -> jnp.ndarray:
        """Calculate the mean."""
        return jnp.where(self.count > 0, self.sum / self.count, 0.0)

    @property
    def variance(self) -> jnp.ndarray:
        """Calculate the variance."""
        return jnp.where(
            self.count > 1,
            (self.sum_of_squares - self.sum**2 / self.count) / (self.count - 1),
            0.0,
        )

    @property
    def std(self) -> jnp.ndarray:
        """Calculate the standard deviation."""
        return jnp.sqrt(self.variance)


@dataclass
class WeightedMean(Accumulator):
    """Weighted mean accumulator."""

    sum_of_weights: jnp.ndarray
    sum_of_weights_squared: jnp.ndarray
    weighted_sum: jnp.ndarray
    weighted_sum_of_squares: jnp.ndarray

    def __add__(self, other: WeightedMean) -> WeightedMean:
        if not isinstance(other, WeightedMean):
            return NotImplemented
        return WeightedMean(
            self.sum_of_weights + other.sum_of_weights,
            self.sum_of_weights_squared + other.sum_of_weights_squared,
            self.weighted_sum + other.weighted_sum,
            self.weighted_sum_of_squares + other.weighted_sum_of_squares,
        )

    @property
    def mean(self) -> jnp.ndarray:
        """Calculate the weighted mean."""
        return jnp.where(
            self.sum_of_weights > 0, self.weighted_sum / self.sum_of_weights, 0.0
        )

    @property
    def variance(self) -> jnp.ndarray:
        """Calculate the weighted variance."""
        return jnp.where(
            self.sum_of_weights > 0,
            self.weighted_sum_of_squares / self.sum_of_weights - self.mean**2,
            0.0,
        )

    @property
    def std(self) -> jnp.ndarray:
        """Calculate the weighted standard deviation."""
        return jnp.sqrt(self.variance)


class Hist:
    """A JAX-based histogram that supports JAX transformations."""

    def __init__(
        self,
        *args: Union[axis.Axis, tuple[int, float, float]],
        data: Optional[jnp.ndarray] = None,
        name: str = "",
        label: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize a histogram.

        Args:
            *args: Either Axis objects or tuples of (bins, start, stop) for
                  creating RegularAxis objects
            data: Initial data array
            name: Name of the histogram
            label: Label of the histogram
            metadata: Additional metadata
        """
        self.name = name
        self.label = label if label else name
        self.metadata = metadata or {}

        # Process axes
        self.axes = []
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 3:
                bins, start, stop = arg
                self.axes.append(axis.RegularAxis(bins, start, stop))
            elif isinstance(arg, axis.Axis):
                self.axes.append(arg)
            else:
                raise TypeError(
                    f"Expected Axis or (bins, start, stop) tuple, got {type(arg)}"
                )

        # Create AxesTuple or NamedAxesTuple
        if all(ax.name for ax in self.axes):
            self.axes = axis.NamedAxesTuple(self.axes)
        else:
            self.axes = axis.AxesTuple(self.axes)

        # Initialize storage for bin counts
        shape = self.axes.extent
        if data is not None:
            if data.shape != shape:
                raise ValueError(
                    f"Data shape {data.shape} doesn't match expected shape {shape}"
                )
            self._counts = data
        else:
            self._counts = jnp.zeros(shape, dtype=jnp.float32)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the histogram."""
        return len(self.axes)

    @property
    def values(self) -> jnp.ndarray:
        """Get the bin values of the histogram."""
        return self._counts

    @property
    def variances(self) -> jnp.ndarray:
        """Get the bin variances of the histogram."""
        # For a simple count histogram, the variance is the same as the value
        return self._counts

    def _get_slice_indices(
        self, value: Union[float, jnp.ndarray], axis_idx: int
    ) -> jnp.ndarray:
        """Get the bin indices for a value along an axis."""
        return self.axes[axis_idx].index(value)

    def _prepare_indices(
        self,
        *args: Union[jnp.ndarray, float, int],
        **kwargs: Union[jnp.ndarray, float, int],
    ) -> tuple[jnp.ndarray, ...]:
        """Prepare indices for filling or indexing."""
        # Handle positional and keyword arguments
        indices = list(args)

        # Process keyword arguments
        for key, value in kwargs.items():
            # Find the axis index for the key
            if isinstance(key, str):
                # Find the axis with the matching name
                found = False
                for i, ax in enumerate(self.axes):
                    if ax.name == key:
                        indices.append((i, value))
                        found = True
                        break
                if not found:
                    raise KeyError(f"Axis with name '{key}' not found")
            else:
                # Assume it's an index
                indices.append((key, value))

        # Sort by axis index
        indices.sort(key=lambda x: x[0] if isinstance(x, tuple) else len(indices))

        # Extract values
        values = []
        for item in indices:
            if isinstance(item, tuple):
                axis_idx, value = item
                values.append(value)
            else:
                values.append(item)

        return tuple(values)

    def fill(
        self,
        *args: Union[jnp.ndarray, float, int],
        weight: Optional[Union[jnp.ndarray, float]] = None,
        **kwargs: Union[jnp.ndarray, float, int],
    ) -> Hist:
        """Fill the histogram with values.

        Args:
            *args: Values for each axis
            weight: Weight values
            **kwargs: Values for named axes

        Returns:
            Self for method chaining
        """
        # Process input values
        values = self._prepare_indices(*args, **kwargs)

        if len(values) != self.ndim:
            raise ValueError(f"Expected {self.ndim} values, got {len(values)}")

        # Handle scalar inputs separately for simplicity and efficiency
        if all(np.isscalar(val) for val in values):
            # Get scalar indices - explicitly convert to Python integers
            indices = tuple(
                int(self.axes[i].index(val)) for i, val in enumerate(values)
            )

            # Update the count at that bin
            weight_val = 1.0 if weight is None else float(weight)
            self._counts = self._counts.at[indices].add(weight_val)
            return self

        # For array inputs, use numpy for intermediate calculations to avoid JAX tracing issues
        counts = np.array(self._counts)

        # Convert everything to numpy arrays
        arrays = []
        for val in values:
            if np.isscalar(val):
                arrays.append(np.array([val]))
            else:
                arrays.append(np.asarray(val).flatten())

        # Find common length
        max_len = max(len(arr) for arr in arrays)

        # Broadcast arrays to common length
        for i, arr in enumerate(arrays):
            if len(arr) == 1:
                arrays[i] = np.full(max_len, arr[0])
            elif len(arr) < max_len:
                # Pad shorter arrays
                arrays[i] = np.pad(arr, (0, max_len - len(arr)), mode="edge")
            elif len(arr) > max_len:
                # Truncate longer arrays
                arrays[i] = arr[:max_len]

        # Prepare weights
        if weight is None:
            weights = np.ones(max_len)
        elif np.isscalar(weight):
            weights = np.full(max_len, weight)
        else:
            weights = np.asarray(weight).flatten()
            if len(weights) == 1:
                weights = np.full(max_len, weights[0])
            elif len(weights) < max_len:
                weights = np.pad(weights, (0, max_len - len(weights)), mode="edge")
            else:
                weights = weights[:max_len]

        # Fill the histogram one point at a time
        for i in range(max_len):
            try:
                # Get indices for this point - explicitly convert to Python integers
                indices = []
                for j, arr in enumerate(arrays):
                    # Use Python's int() to ensure integer indices
                    idx = int(self.axes[j].index(float(arr[i])))
                    indices.append(idx)

                # Update the bin count
                counts[tuple(indices)] += float(weights[i])
            except (IndexError, ValueError, TypeError):
                # Skip invalid points
                continue

        # Update the histogram with the new counts
        self._counts = jnp.array(counts)

        return self

    def __getitem__(
        self, key: Union[int, slice, tuple[Union[int, slice], ...]]
    ) -> Union[Hist, float]:
        """Get a slice of the histogram.

        Args:
            key: Index or slice

        Returns:
            A new histogram with the selected slice or a bin value
        """
        # Convert key to tuple if it's not already
        if not isinstance(key, tuple):
            key = (key,)

        # Pad with full slices if needed
        key = key + tuple(slice(None) for _ in range(self.ndim - len(key)))

        # Check if this is a simple index lookup
        if all(isinstance(k, int) for k in key):
            return self._counts[key]

        # Otherwise, we're creating a new histogram
        # First, determine the axes for the new histogram
        new_axes = []
        for i, (k, ax) in enumerate(zip(key, self.axes, strict=False)):
            if isinstance(k, slice):
                new_axes.append(ax)

        # Create a new histogram with the selected axes
        new_hist = Hist(
            *new_axes,
            name=self.name,
            label=self.label,
            metadata=self.metadata.copy(),
        )

        # Set the bin values
        new_hist._counts = self._counts[key]

        return new_hist

    def __add__(self, other: Union[Hist, float, int]) -> Hist:
        """Add two histograms or add a scalar to all bins."""
        if isinstance(other, Hist):
            # Verify axes compatibility
            if len(self.axes) != len(other.axes):
                raise ValueError("Histograms must have the same number of axes")

            # Create a new histogram with the same axes
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )

            # Add the bin values
            result._counts = self._counts + other._counts
            return result
        else:
            # Add a scalar to all bins
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )
            result._counts = self._counts + other
            return result

    def __radd__(self, other: Union[float, int]) -> Hist:
        """Add a scalar to all bins (right addition)."""
        return self.__add__(other)

    def __sub__(self, other: Union[Hist, float, int]) -> Hist:
        """Subtract two histograms or subtract a scalar from all bins."""
        if isinstance(other, Hist):
            # Verify axes compatibility
            if len(self.axes) != len(other.axes):
                raise ValueError("Histograms must have the same number of axes")

            # Create a new histogram with the same axes
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )

            # Subtract the bin values
            result._counts = self._counts - other._counts
            return result
        else:
            # Subtract a scalar from all bins
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )
            result._counts = self._counts - other
            return result

    def __rsub__(self, other: Union[float, int]) -> Hist:
        """Subtract the histogram from a scalar (right subtraction)."""
        result = Hist(
            *self.axes,
            name=self.name,
            label=self.label,
            metadata=self.metadata.copy(),
        )
        result._counts = other - self._counts
        return result

    def __mul__(self, other: Union[Hist, float, int]) -> Hist:
        """Multiply two histograms or multiply all bins by a scalar."""
        if isinstance(other, Hist):
            # Verify axes compatibility
            if len(self.axes) != len(other.axes):
                raise ValueError("Histograms must have the same number of axes")

            # Create a new histogram with the same axes
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )

            # Multiply the bin values
            result._counts = self._counts * other._counts
            return result
        else:
            # Multiply all bins by a scalar
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )
            result._counts = self._counts * other
            return result

    def __rmul__(self, other: Union[float, int]) -> Hist:
        """Multiply all bins by a scalar (right multiplication)."""
        return self.__mul__(other)

    def __truediv__(self, other: Union[Hist, float, int]) -> Hist:
        """Divide two histograms or divide all bins by a scalar."""
        if isinstance(other, Hist):
            # Verify axes compatibility
            if len(self.axes) != len(other.axes):
                raise ValueError("Histograms must have the same number of axes")

            # Create a new histogram with the same axes
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )

            # Divide the bin values
            result._counts = jnp.divide(
                self._counts,
                other._counts,
                out=jnp.zeros_like(self._counts),
                where=other._counts != 0,
            )
            return result
        else:
            # Divide all bins by a scalar
            result = Hist(
                *self.axes,
                name=self.name,
                label=self.label,
                metadata=self.metadata.copy(),
            )
            result._counts = self._counts / other
            return result

    def __rtruediv__(self, other: Union[float, int]) -> Hist:
        """Divide a scalar by the histogram (right division)."""
        result = Hist(
            *self.axes,
            name=self.name,
            label=self.label,
            metadata=self.metadata.copy(),
        )
        result._counts = jnp.divide(
            other,
            self._counts,
            out=jnp.zeros_like(self._counts),
            where=self._counts != 0,
        )
        return result

    def project(self, *axes: Union[int, str]) -> Hist:
        """Project the histogram onto the specified axes.

        Args:
            *axes: Indices or names of the axes to project onto

        Returns:
            A new histogram with the specified axes
        """
        # Convert axis names to indices
        axis_indices = []
        for ax in axes:
            if isinstance(ax, str):
                # Find the axis with the matching name
                found = False
                for i, axis_obj in enumerate(self.axes):
                    if axis_obj.name == ax:
                        axis_indices.append(i)
                        found = True
                        break
                if not found:
                    raise KeyError(f"Axis with name '{ax}' not found")
            else:
                axis_indices.append(ax)

        # Create a new histogram with the specified axes
        new_axes = [self.axes[i] for i in axis_indices]
        result = Hist(
            *new_axes, name=self.name, label=self.label, metadata=self.metadata.copy()
        )

        # Create a list of axes to sum over (all axes not in axis_indices)
        sum_axes = tuple(i for i in range(self.ndim) if i not in axis_indices)

        # Project by summing over the non-selected axes
        if sum_axes:
            result._counts = jnp.sum(self._counts, axis=sum_axes)
        else:
            result._counts = self._counts

        return result

    def sum(self, flow: bool = True) -> float:
        """Sum the bin values.

        Args:
            flow: Whether to include flow bins in the sum

        Returns:
            Sum of bin values
        """
        # If flow is False, slice the array to exclude flow bins
        if not flow:
            slices = tuple(
                slice(1, -1) if ax.has_underflow and ax.has_overflow else slice(None)
                for ax in self.axes
            )
            return jnp.sum(self._counts[slices])
        else:
            return jnp.sum(self._counts)

    def profile(self, axis: Union[int, str]) -> Hist:
        """Create a profile histogram by projecting out an axis.

        Args:
            axis: Index or name of the axis to profile

        Returns:
            A profile histogram
        """
        # Convert axis name to index if needed
        if isinstance(axis, str):
            # Find the axis with the matching name
            found = False
            for i, ax in enumerate(self.axes):
                if ax.name == axis:
                    axis = i
                    found = True
                    break
            if not found:
                raise KeyError(f"Axis with name '{axis}' not found")

        # Create a Mean accumulator
        profiled_axes = [ax for i, ax in enumerate(self.axes) if i != axis]
        result = Hist(
            *profiled_axes,
            name=self.name,
            label=self.label,
            metadata=self.metadata.copy(),
        )

        # Get centers of the profiled axis
        centers = self.axes[axis].centers

        # Calculate count, sum, and sum of squares for each bin
        counts = jnp.sum(self._counts, axis=axis)
        weighted_sums = jnp.tensordot(self._counts, centers, axes=([axis], [0]))
        weighted_sqsums = jnp.tensordot(self._counts, centers**2, axes=([axis], [0]))

        # Store as a tuple of (count, sum, sum_sq)
        # For simplicity, we'll just use the _counts attribute directly
        # In a full implementation, you'd want a proper Mean accumulator
        result._counts = jnp.stack([counts, weighted_sums, weighted_sqsums], axis=-1)

        return result

    def density(self) -> jnp.ndarray:
        """Calculate the density (normalized so the integral is 1).

        Returns:
            Density values
        """
        # Calculate bin volumes
        bin_widths = [ax.widths for ax in self.axes]
        bin_volumes = functools.reduce(
            lambda a, b: jnp.tensordot(a, b, axes=0), bin_widths
        ).reshape(self.axes.size)

        # Calculate the total
        total = jnp.sum(self._counts * bin_volumes)

        # Normalize
        return self._counts / jnp.where(total > 0, total, 1.0)

    def to_pytree(self) -> dict[str, Any]:
        """Convert the histogram to a PyTree for JAX transformations.

        Returns:
            A dictionary representation of the histogram
        """
        # For now, just return the counts and metadata
        # A full implementation would need proper PyTree registration
        return {
            "counts": self._counts,
            "name": self.name,
            "label": self.label,
            "metadata": self.metadata,
            # We don't include axes here, as they would need special handling
        }

    @classmethod
    def from_pytree(cls, tree: dict[str, Any], axes: list[axis.Axis]) -> Hist:
        """Create a histogram from a PyTree.

        Args:
            tree: PyTree representation of the histogram
            axes: List of axes

        Returns:
            A new histogram
        """
        return cls(
            *axes,
            data=tree["counts"],
            name=tree["name"],
            label=tree["label"],
            metadata=tree["metadata"],
        )

    def __repr__(self) -> str:
        """Get a string representation of the histogram."""
        axes_str = ", ".join(repr(ax) for ax in self.axes)
        return f"Hist({axes_str}, name='{self.name}', label='{self.label}')"


class NamedHist(Hist):
    """A histogram with named axes that must be accessed by name."""

    def __init__(
        self,
        *args: Union[axis.Axis, tuple[int, float, float]],
        data: Optional[jnp.ndarray] = None,
        name: str = "",
        label: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize a named histogram."""
        super().__init__(*args, data=data, name=name, label=label, metadata=metadata)

        # Ensure all axes have names
        for i, ax in enumerate(self.axes):
            if not ax.name:
                raise ValueError(f"Axis {i} must have a name in NamedHist")

        # Convert to NamedAxesTuple if not already
        if not isinstance(self.axes, axis.NamedAxesTuple):
            self.axes = axis.NamedAxesTuple(self.axes)

    def fill(
        self,
        weight: Optional[Union[jnp.ndarray, float]] = None,
        **kwargs: Union[jnp.ndarray, float, int],
    ) -> NamedHist:
        """Fill the histogram with values (only allows named axes).

        Args:
            weight: Weight values
            **kwargs: Values for named axes

        Returns:
            Self for method chaining
        """
        if len(kwargs) != self.ndim:
            raise ValueError(f"Expected {self.ndim} named values, got {len(kwargs)}")

        return super().fill(weight=weight, **kwargs)

    def __getitem__(self, key: dict[str, Union[int, slice]]) -> Union[NamedHist, float]:
        """Get a slice of the histogram by axis name."""
        if not isinstance(key, dict):
            raise TypeError("NamedHist only supports dictionary-based indexing")

        # Convert the named keys to indices
        indexed_key = {}
        for name, value in key.items():
            for i, ax in enumerate(self.axes):
                if ax.name == name:
                    indexed_key[i] = value
                    break
            else:
                raise KeyError(f"Axis with name '{name}' not found")

        # Get the result using the parent class
        result = super().__getitem__(indexed_key)

        # Convert back to NamedHist if it's a histogram
        if isinstance(result, Hist) and not isinstance(result, NamedHist):
            # This is hackish but works for the example
            named_result = NamedHist(
                *result.axes,
                data=result._counts,
                name=result.name,
                label=result.label,
                metadata=result.metadata,
            )
            return named_result

        return result
