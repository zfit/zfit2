"""
Axis module for jax-hist.

This module provides various axis types for histograms, including
Regular, Variable, Integer, and Category axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, TypeVar, Union, cast

import jax
import jax.numpy as jnp
import numpy as np


class AxisTraits:
    """Traits for axes."""
    
    def __init__(
        self, 
        ordered: bool = True, 
        discrete: bool = False, 
        circular: bool = False
    ):
        self.ordered = ordered
        self.discrete = discrete
        self.circular = circular
    
    def __repr__(self) -> str:
        traits = []
        if self.ordered:
            traits.append("ordered")
        if self.discrete:
            traits.append("discrete")
        if self.circular:
            traits.append("circular")
        return f"AxisTraits({', '.join(traits)})"


class OverflowBehavior(Enum):
    """Behavior for values outside the axis range."""
    NONE = auto()    # No overflow/underflow bins
    OVERFLOW = auto()    # Only overflow bin
    UNDERFLOW = auto()   # Only underflow bin
    BOTH = auto()    # Both overflow and underflow bins
    

@dataclass
class Metadata:
    """Metadata for an axis."""
    name: str = ""
    label: str = ""
    # Allow arbitrary metadata storage
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        if not self.label:
            self.label = self.name


class Axis:
    """Base class for all axes."""
    
    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: OverflowBehavior = OverflowBehavior.BOTH,
    ):
        self._metadata = Metadata(name=name, label=label, extra=metadata or {})
        self._overflow = overflow
        self.traits = AxisTraits()
        
    @property
    def name(self) -> str:
        """Get the name of the axis."""
        return self._metadata.name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the axis."""
        self._metadata.name = value
    
    @property
    def label(self) -> str:
        """Get the label of the axis."""
        return self._metadata.label or self._metadata.name
    
    @label.setter
    def label(self, value: str) -> None:
        """Set the label of the axis."""
        self._metadata.label = value
    
    @property
    def has_underflow(self) -> bool:
        """Whether this axis has an underflow bin."""
        return self._overflow in (OverflowBehavior.UNDERFLOW, OverflowBehavior.BOTH)
    
    @property
    def has_overflow(self) -> bool:
        """Whether this axis has an overflow bin."""
        return self._overflow in (OverflowBehavior.OVERFLOW, OverflowBehavior.BOTH)
    
    @property
    def n_bins(self) -> int:
        """Get the number of bins in the axis."""
        raise NotImplementedError("Subclasses must implement n_bins")
    
    @property
    def edges(self) -> jnp.ndarray:
        """Get the bin edges of the axis."""
        raise NotImplementedError("Subclasses must implement edges")
    
    @property
    def centers(self) -> jnp.ndarray:
        """Get the bin centers of the axis."""
        edges = self.edges
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def widths(self) -> jnp.ndarray:
        """Get the bin widths of the axis."""
        edges = self.edges
        return edges[1:] - edges[:-1]
    
    def index(self, value: Union[jnp.ndarray, float]) -> jnp.ndarray:
        """Get the bin index for a value."""
        raise NotImplementedError("Subclasses must implement index")
    
    def indices(self, values: jnp.ndarray) -> jnp.ndarray:
        """Get the bin indices for an array of values."""
        return jax.vmap(self.index)(values)
    
    def __len__(self) -> int:
        """Get the number of bins in the axis."""
        return self.n_bins


class RegularAxis(Axis):
    """A regularly spaced axis."""
    
    def __init__(
        self,
        bins: int,
        start: float,
        stop: float,
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: OverflowBehavior = OverflowBehavior.BOTH,
        circular: bool = False,
    ):
        """Initialize a regular axis.
        
        Args:
            bins: Number of bins
            start: Start value (included)
            stop: Stop value (included in the last bin)
            name: Name of the axis
            label: Label of the axis
            metadata: Additional metadata
            overflow: Overflow handling behavior
            circular: Whether the axis is circular
        """
        super().__init__(name=name, label=label, metadata=metadata, overflow=overflow)
        self._bins = bins
        self._start = start
        self._stop = stop
        self.traits = AxisTraits(ordered=True, discrete=False, circular=circular)
    
    @property
    def n_bins(self) -> int:
        """Get the number of bins in the axis."""
        return self._bins
    
    @property
    def extent(self) -> int:
        """Get the total number of bins including flow bins."""
        return self._bins + self.has_underflow + self.has_overflow
    
    @property
    def edges(self) -> jnp.ndarray:
        """Get the bin edges of the axis."""
        return jnp.linspace(self._start, self._stop, self._bins + 1)


    # Updated RegularAxis.index method
    def index(self, value: Union[jnp.ndarray, float]) -> jnp.ndarray:
        """Get the bin index for a value."""
        # Convert to float for JAX handling
        value = jnp.asarray(value, dtype=jnp.float32)

        # Calculate the bin index
        bin_width = (self._stop - self._start) / self._bins
        raw_index = (value - self._start) / bin_width

        # Handle circular axes
        if self.traits.circular:
            raw_index = raw_index % self._bins

        # Round down to get the bin index - explicitly cast to int32
        index = jnp.floor(raw_index).astype(jnp.int32)

        # Handle underflow/overflow
        index = jnp.where(
            jnp.logical_and(index >= 0, index < self._bins),
            index,
            jnp.where(
                index < 0,
                # Underflow handling
                jnp.where(
                    self.has_underflow,
                    jnp.array(-1, dtype=jnp.int32),  # Underflow bin
                    jnp.zeros_like(index, dtype=jnp.int32)  # First bin
                ),
                # Overflow handling
                jnp.where(
                    self.has_overflow,
                    jnp.array(self._bins, dtype=jnp.int32),  # Overflow bin
                    jnp.array(self._bins - 1, dtype=jnp.int32)  # Last bin
                )
            )
        )

        return index
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        return (
            f"RegularAxis(bins={self._bins}, start={self._start}, stop={self._stop}, "
            f"name='{self.name}', label='{self.label}')"
        )


class VariableAxis(Axis):
    """An axis with variable bin widths."""
    
    def __init__(
        self,
        edges: Union[List[float], np.ndarray, jnp.ndarray],
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: OverflowBehavior = OverflowBehavior.BOTH,
        circular: bool = False,
    ):
        """Initialize a variable axis.
        
        Args:
            edges: Bin edges
            name: Name of the axis
            label: Label of the axis
            metadata: Additional metadata
            overflow: Overflow handling behavior
            circular: Whether the axis is circular
        """
        super().__init__(name=name, label=label, metadata=metadata, overflow=overflow)
        self._edges = jnp.asarray(edges, dtype=jnp.float32)
        self.traits = AxisTraits(ordered=True, discrete=False, circular=circular)
    
    @property
    def n_bins(self) -> int:
        """Get the number of bins in the axis."""
        return len(self._edges) - 1
    
    @property
    def extent(self) -> int:
        """Get the total number of bins including flow bins."""
        return self.n_bins + self.has_underflow + self.has_overflow
    
    @property
    def edges(self) -> jnp.ndarray:
        """Get the bin edges of the axis."""
        return self._edges

    def index(self, value: Union[jnp.ndarray, float]) -> jnp.ndarray:
        """Get the bin index for a value."""
        value = jnp.asarray(value, dtype=jnp.float32)

        # Use searchsorted to find the bin index
        raw_index = jnp.searchsorted(self._edges, value) - 1

        # Ensure int32 type
        raw_index = raw_index.astype(jnp.int32)

        # Handle circular axes
        if self.traits.circular:
            raw_index = raw_index % self.n_bins

        # Handle underflow/overflow
        index = jnp.where(
            jnp.logical_and(raw_index >= 0, raw_index < self.n_bins),
            raw_index,
            jnp.where(
                raw_index < 0,
                # Underflow handling
                jnp.where(
                    self.has_underflow,
                    jnp.array(-1, dtype=jnp.int32),  # Underflow bin
                    jnp.zeros_like(raw_index, dtype=jnp.int32)  # First bin
                ),
                # Overflow handling
                jnp.where(
                    self.has_overflow,
                    jnp.array(self.n_bins, dtype=jnp.int32),  # Overflow bin
                    jnp.array(self.n_bins - 1, dtype=jnp.int32)  # Last bin
                )
            )
        )

        return index
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        edges_str = f"[{self._edges[0]}, ..., {self._edges[-1]}]"
        return (
            f"VariableAxis(edges={edges_str}, n_bins={self.n_bins}, "
            f"name='{self.name}', label='{self.label}')"
        )


class IntegerAxis(Axis):
    """An axis for integer values."""
    
    def __init__(
        self,
        start: int,
        stop: int,
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: OverflowBehavior = OverflowBehavior.BOTH,
    ):
        """Initialize an integer axis.
        
        Args:
            start: Start value (included)
            stop: Stop value (included)
            name: Name of the axis
            label: Label of the axis
            metadata: Additional metadata
            overflow: Overflow handling behavior
        """
        super().__init__(name=name, label=label, metadata=metadata, overflow=overflow)
        self._start = start
        self._stop = stop
        self.traits = AxisTraits(ordered=True, discrete=True)
    
    @property
    def n_bins(self) -> int:
        """Get the number of bins in the axis."""
        return self._stop - self._start + 1
    
    @property
    def extent(self) -> int:
        """Get the total number of bins including flow bins."""
        return self.n_bins + self.has_underflow + self.has_overflow
    
    @property
    def edges(self) -> jnp.ndarray:
        """Get the bin edges of the axis."""
        # For integer axis, the edges are at integer values - 0.5 and stop + 0.5
        return jnp.arange(self._start - 0.5, self._stop + 1.5, 1.0)
    
    @property
    def centers(self) -> jnp.ndarray:
        """Get the bin centers of the axis."""
        return jnp.arange(self._start, self._stop + 1, 1)
    
    def index(self, value: Union[jnp.ndarray, int, float]) -> jnp.ndarray:
        """Get the bin index for a value."""
        value = jnp.asarray(value, dtype=jnp.int32)
        
        # Calculate the bin index
        raw_index = value - self._start
        
        # Handle underflow/overflow
        index = jnp.where(
            jnp.logical_and(value >= self._start, value <= self._stop),
            raw_index,
            jnp.where(
                value < self._start,
                # Underflow handling
                jnp.where(
                    self.has_underflow,
                    -1,  # Underflow bin
                    jnp.zeros_like(raw_index)  # First bin
                ),
                # Overflow handling
                jnp.where(
                    self.has_overflow,
                    self.n_bins,  # Overflow bin
                    self.n_bins - 1  # Last bin
                )
            )
        )
        
        return index
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        return (
            f"IntegerAxis(start={self._start}, stop={self._stop}, "
            f"name='{self.name}', label='{self.label}')"
        )


# Type for categories
T = TypeVar('T', str, int)

class CategoryAxis(Generic[T]):
    """An axis for categorical data (strings or integers)."""
    
    def __init__(
        self,
        categories: List[T],
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: bool = True,
        growth: bool = False,
    ):
        """Initialize a category axis.
        
        Args:
            categories: List of categories
            name: Name of the axis
            label: Label of the axis
            metadata: Additional metadata
            overflow: Whether to include an overflow bin
            growth: Whether to allow growing categories
        """
        self._metadata = Metadata(name=name, label=label, extra=metadata or {})
        self._categories = categories
        self._overflow = overflow
        self._growth = growth
        self.traits = AxisTraits(ordered=False, discrete=True)
        
        # Create a mapping from category to index
        self._category_map = {cat: i for i, cat in enumerate(categories)}
    
    @property
    def name(self) -> str:
        """Get the name of the axis."""
        return self._metadata.name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the axis."""
        self._metadata.name = value
    
    @property
    def label(self) -> str:
        """Get the label of the axis."""
        return self._metadata.label or self._metadata.name
    
    @label.setter
    def label(self, value: str) -> None:
        """Set the label of the axis."""
        self._metadata.label = value
    
    @property
    def has_overflow(self) -> bool:
        """Whether this axis has an overflow bin."""
        return self._overflow
    
    @property
    def categories(self) -> List[T]:
        """Get the categories of the axis."""
        return self._categories
    
    @property
    def n_bins(self) -> int:
        """Get the number of bins in the axis."""
        return len(self._categories)
    
    @property
    def extent(self) -> int:
        """Get the total number of bins including flow bins."""
        return self.n_bins + self.has_overflow
    
    def index(self, value: T) -> jnp.ndarray:
        """Get the bin index for a value."""
        # Non-JAX operation to look up the index
        # Later we'll need a more JAX-friendly approach
        idx = self._category_map.get(value, -1)
        
        if idx == -1:
            if self._growth:
                # Add the category (this is a non-JAX side effect)
                idx = len(self._categories)
                self._categories.append(value)
                self._category_map[value] = idx
            elif self.has_overflow:
                # Use overflow bin
                idx = self.n_bins
            else:
                # Invalid index
                idx = 0  # Default to first bin
        
        return jnp.array(idx, dtype=jnp.int32)
    
    def indices(self, values: List[T]) -> jnp.ndarray:
        """Get the bin indices for a list of values."""
        return jnp.array([self.index(v).item() for v in values], dtype=jnp.int32)
    
    def __len__(self) -> int:
        """Get the number of bins in the axis."""
        return self.n_bins
    
    def __iter__(self):
        """Iterate over the categories."""
        return iter(self._categories)
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        cats_str = str(self._categories[:3])
        if len(self._categories) > 3:
            cats_str = cats_str[:-1] + ", ...]"
        return (
            f"CategoryAxis(categories={cats_str}, "
            f"name='{self.name}', label='{self.label}')"
        )


class StrCategoryAxis(CategoryAxis[str]):
    """An axis for string categories."""
    
    def __init__(
        self,
        categories: List[str],
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: bool = True,
        growth: bool = False,
    ):
        """Initialize a string category axis."""
        super().__init__(
            categories=categories,
            name=name,
            label=label,
            metadata=metadata,
            overflow=overflow,
            growth=growth,
        )
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        cats_str = str(self._categories[:3])
        if len(self._categories) > 3:
            cats_str = cats_str[:-1] + ", ...]"
        return (
            f"StrCategoryAxis(categories={cats_str}, "
            f"name='{self.name}', label='{self.label}')"
        )


class IntCategoryAxis(CategoryAxis[int]):
    """An axis for integer categories."""
    
    def __init__(
        self,
        categories: List[int],
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        overflow: bool = True,
        growth: bool = False,
    ):
        """Initialize an integer category axis."""
        super().__init__(
            categories=categories,
            name=name,
            label=label,
            metadata=metadata,
            overflow=overflow,
            growth=growth,
        )
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        cats_str = str(self._categories[:3])
        if len(self._categories) > 3:
            cats_str = cats_str[:-1] + ", ...]"
        return (
            f"IntCategoryAxis(categories={cats_str}, "
            f"name='{self.name}', label='{self.label}')"
        )


class BooleanAxis(CategoryAxis[bool]):
    """An axis for boolean values."""
    
    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a boolean axis."""
        super().__init__(
            categories=[False, True],
            name=name,
            label=label,
            metadata=metadata,
            overflow=False,
            growth=False,
        )
    
    def __repr__(self) -> str:
        """Get a string representation of the axis."""
        return f"BooleanAxis(name='{self.name}', label='{self.label}')"


# Convenient aliases matching the hist library
Regular = RegularAxis
Variable = VariableAxis
Integer = IntegerAxis
IntCategory = IntCategoryAxis
StrCategory = StrCategoryAxis
Boolean = BooleanAxis


class AxesTuple:
    """A tuple of axes."""
    
    def __init__(self, axes):
        self.axes = list(axes)
    
    def __getitem__(self, idx):
        return self.axes[idx]
    
    def __len__(self):
        return len(self.axes)
    
    def __iter__(self):
        return iter(self.axes)
    
    @property
    def size(self):
        """Get the shape of the histogram (excluding flow bins)."""
        return tuple(axis.n_bins for axis in self.axes)
    
    @property
    def extent(self):
        """Get the shape of the histogram (including flow bins)."""
        return tuple(axis.extent for axis in self.axes)
    
    @property
    def edges(self):
        """Get the bin edges of all axes."""
        return tuple(axis.edges for axis in self.axes)
    
    @property
    def centers(self):
        """Get the bin centers of all axes."""
        return tuple(axis.centers for axis in self.axes)
    
    @property
    def widths(self):
        """Get the bin widths of all axes."""
        return tuple(axis.widths for axis in self.axes)
    
    @property
    def name(self):
        """Get the names of all axes."""
        return tuple(axis.name for axis in self.axes)
    
    @property
    def label(self):
        """Get the labels of all axes."""
        return tuple(axis.label for axis in self.axes)


class NamedAxesTuple(AxesTuple):
    """A tuple of named axes with lookup by name."""
    
    def __init__(self, axes):
        super().__init__(axes)
        # Validate that all axes have names
        for i, axis in enumerate(self.axes):
            if not axis.name:
                raise ValueError(f"Axis {i} must have a name")
        
        # Check for duplicates
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate axis names are not allowed")
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            for axis in self.axes:
                if axis.name == idx:
                    return axis
            raise KeyError(f"Axis with name '{idx}' not found")
        return super().__getitem__(idx)
