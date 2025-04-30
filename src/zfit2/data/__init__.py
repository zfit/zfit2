"""
jax-hist: A JAX-based histogram library.

This library provides histogram functionality built on JAX,
allowing for automatic differentiation and accelerated computation.
It follows the Unified Histogram Interface (UHI) requirements
and is compatible with JAX transformations like jit, grad, and vmap.
"""

from __future__ import annotations

from . import axis
from .axis import (
    AxesTuple,
    Axis,
    CategoryAxis,
    NamedAxesTuple,
)
from .axis import (
    BooleanAxis as Boolean,
)
from .axis import (
    IntCategoryAxis as IntCategory,
)
from .axis import (
    IntegerAxis as Integer,
)
from .axis import (
    RegularAxis as Regular,
)
from .axis import (
    StrCategoryAxis as StrCategory,
)
from .axis import (
    VariableAxis as Variable,
)
from .histogram import (
    Accumulator,
    Count,
    Hist,
    Mean,
    NamedHist,
    WeightedMean,
    WeightedSum,
)
from .utils import (
    grad_bin_count,
    hist,
    histogram2d,
    histogramdd,
    jit_fill,
    vmap_fill,
)

# Define version
__version__ = "0.1.0"

# Define all exports
__all__ = [
    # Axis types
    "Axis",
    "AxesTuple",
    "Boolean",
    "CategoryAxis",
    "IntCategory",
    "Integer",
    "NamedAxesTuple",
    "Regular",
    "StrCategory",
    "Variable",
    # Histogram classes
    "Accumulator",
    "Count",
    "Hist",
    "Mean",
    "NamedHist",
    "WeightedMean",
    "WeightedSum",
    # Utility functions
    "grad_bin_count",
    "hist",
    "histogram2d",
    "histogramdd",
    "jit_fill",
    "vmap_fill",
    # Version
    "__version__",
]
