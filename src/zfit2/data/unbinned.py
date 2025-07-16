"""Unbinned data container for zfit2.

This module provides a DataFrame-like container for unbinned data,
backed by JAX arrays for efficient computation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class UnbinnedData:
    """Container for unbinned data with DataFrame-like interface.

    This class provides a way to store and manipulate unbinned data with
    named variables (columns) and optional sample weights. It wraps JAX
    arrays for efficient computation while providing a familiar interface
    similar to pandas DataFrames.

    Attributes:
        data: The underlying data as a JAX array (n_samples, n_variables)
        variables: List of variable names (column names)
        weights: Optional sample weights as a JAX array (n_samples,)
    """

    def __init__(
        self,
        data: np.ndarray | jnp.ndarray | dict[str, np.ndarray | jnp.ndarray],
        variables: list[str] | None = None,
        weights: np.ndarray | jnp.ndarray | None = None,
    ):
        """Initialize UnbinnedData.

        Args:
            data: Either a 2D array (n_samples, n_variables) or a dict mapping
                  variable names to 1D arrays
            variables: List of variable names. Required if data is an array.
                      Ignored if data is a dict.
            weights: Optional sample weights (n_samples,)
        """
        if isinstance(data, dict):
            # Convert dict to array format
            if variables is not None:
                msg = "variables argument is ignored when data is a dict"
                raise ValueError(msg)
            self.variables = list(data.keys())
            # Stack arrays column-wise
            arrays = [jnp.asarray(data[var]) for var in self.variables]
            self.data = jnp.column_stack(arrays)
        else:
            # Data is already an array
            self.data = jnp.asarray(data)
            if self.data.ndim == 1:
                self.data = self.data.reshape(-1, 1)
            elif self.data.ndim != 2:
                msg = f"data must be 1D or 2D array, got {self.data.ndim}D"
                raise ValueError(msg)

            if variables is None:
                # Generate default variable names
                n_vars = self.data.shape[1]
                self.variables = [f"var_{i}" for i in range(n_vars)]
            else:
                if len(variables) != self.data.shape[1]:
                    msg = f"Number of variables ({len(variables)}) must match data shape ({self.data.shape[1]})"
                    raise ValueError(msg)
                self.variables = list(variables)

        # Handle weights
        if weights is not None:
            self.weights = jnp.asarray(weights)
            if self.weights.shape != (self.data.shape[0],):
                msg = f"weights shape {self.weights.shape} doesn't match data shape {self.data.shape}"
                raise ValueError(msg)
        else:
            self.weights = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data (n_samples, n_variables)."""
        return self.data.shape

    @property
    def n_samples(self) -> int:
        """Return the number of samples."""
        return self.data.shape[0]

    @property
    def n_variables(self) -> int:
        """Return the number of variables."""
        return self.data.shape[1]

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def __getitem__(
        self, key: str | int | slice | list[str]
    ) -> jnp.ndarray | UnbinnedData:
        """Access data by variable name or index.

        Args:
            key: Variable name(s), index, or slice

        Returns:
            JAX array for single variable, UnbinnedData for multiple variables
        """
        if isinstance(key, str):
            # Single variable by name
            try:
                idx = self.variables.index(key)
                return self.data[:, idx]
            except ValueError:
                msg = f"Variable '{key}' not found"
                raise KeyError(msg)

        elif isinstance(key, int):
            # Single variable by index
            return self.data[:, key]

        elif isinstance(key, slice):
            # Slice of variables
            sliced_data = self.data[:, key]
            sliced_vars = self.variables[key]
            return UnbinnedData(
                sliced_data, variables=sliced_vars, weights=self.weights
            )

        elif isinstance(key, list):
            # List of variable names
            indices = []
            for var in key:
                try:
                    indices.append(self.variables.index(var))
                except ValueError:
                    msg = f"Variable '{var}' not found"
                    raise KeyError(msg)
            selected_data = self.data[:, indices]
            return UnbinnedData(selected_data, variables=key, weights=self.weights)

        else:
            msg = f"Invalid key type: {type(key)}"
            raise TypeError(msg)

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        weights_info = (
            f", weights={self.weights.shape}" if self.weights is not None else ""
        )
        return f"UnbinnedData(shape={self.shape}, variables={self.variables}{weights_info})"

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        lines = [
            f"UnbinnedData with {self.n_samples} samples and {self.n_variables} variables:"
        ]
        lines.append(f"  Variables: {', '.join(self.variables)}")
        if self.weights is not None:
            lines.append("  Weights: yes")

        # Show first few rows like pandas
        if (n_show := min(5, self.n_samples)) > 0:
            lines.append("\n  First few samples:")
            header = "  " + "\t".join(f"{var:>10}" for var in self.variables)
            lines.append(header)
            for i in range(n_show):
                row = "  " + "\t".join(
                    f"{self.data[i, j]:10.4f}" for j in range(self.n_variables)
                )
                lines.append(row)
            if self.n_samples > n_show:
                lines.append(f"  ... ({self.n_samples - n_show} more rows)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, jnp.ndarray]:
        """Convert to dictionary of arrays."""
        return {var: self.data[:, i] for i, var in enumerate(self.variables)}

    # Arithmetic operations
    def __add__(self, other: float | jnp.ndarray | UnbinnedData) -> UnbinnedData:
        """Add a scalar or array to the data."""
        if isinstance(other, UnbinnedData):
            if other.shape != self.shape:
                msg = f"Shape mismatch: {self.shape} vs {other.shape}"
                raise ValueError(msg)
            if other.variables != self.variables:
                msg = f"Variable mismatch: {self.variables} vs {other.variables}"
                raise ValueError(msg)
            new_data = self.data + other.data
            # Use weights from self (arbitrary choice)
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )
        else:
            new_data = self.data + other
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )

    def __sub__(self, other: float | jnp.ndarray | UnbinnedData) -> UnbinnedData:
        """Subtract a scalar or array from the data."""
        if isinstance(other, UnbinnedData):
            if other.shape != self.shape:
                msg = f"Shape mismatch: {self.shape} vs {other.shape}"
                raise ValueError(msg)
            if other.variables != self.variables:
                msg = f"Variable mismatch: {self.variables} vs {other.variables}"
                raise ValueError(msg)
            new_data = self.data - other.data
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )
        else:
            new_data = self.data - other
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )

    def __mul__(self, other: float | jnp.ndarray | UnbinnedData) -> UnbinnedData:
        """Multiply the data by a scalar or array."""
        if isinstance(other, UnbinnedData):
            if other.shape != self.shape:
                msg = f"Shape mismatch: {self.shape} vs {other.shape}"
                raise ValueError(msg)
            if other.variables != self.variables:
                msg = f"Variable mismatch: {self.variables} vs {other.variables}"
                raise ValueError(msg)
            new_data = self.data * other.data
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )
        else:
            new_data = self.data * other
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )

    def __truediv__(self, other: float | jnp.ndarray | UnbinnedData) -> UnbinnedData:
        """Divide the data by a scalar or array."""
        if isinstance(other, UnbinnedData):
            if other.shape != self.shape:
                msg = f"Shape mismatch: {self.shape} vs {other.shape}"
                raise ValueError(msg)
            if other.variables != self.variables:
                msg = f"Variable mismatch: {self.variables} vs {other.variables}"
                raise ValueError(msg)
            new_data = self.data / other.data
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )
        else:
            new_data = self.data / other
            return UnbinnedData(
                new_data, variables=self.variables, weights=self.weights
            )

    def __pow__(self, other: float | jnp.ndarray) -> UnbinnedData:
        """Raise the data to a power."""
        new_data = self.data**other
        return UnbinnedData(new_data, variables=self.variables, weights=self.weights)

    # Reverse operations
    def __radd__(self, other: float | jnp.ndarray) -> UnbinnedData:
        """Right addition."""
        return self.__add__(other)

    def __rsub__(self, other: float | jnp.ndarray) -> UnbinnedData:
        """Right subtraction."""
        new_data = other - self.data
        return UnbinnedData(new_data, variables=self.variables, weights=self.weights)

    def __rmul__(self, other: float | jnp.ndarray) -> UnbinnedData:
        """Right multiplication."""
        return self.__mul__(other)

    def __rtruediv__(self, other: float | jnp.ndarray) -> UnbinnedData:
        """Right division."""
        new_data = other / self.data
        return UnbinnedData(new_data, variables=self.variables, weights=self.weights)


# JAX PyTree registration
def _unbinned_data_flatten(data_obj: UnbinnedData) -> tuple[list, dict]:
    """Flatten UnbinnedData for JAX PyTree."""
    # Dynamic values
    children = [data_obj.data]
    if data_obj.weights is not None:
        children.append(data_obj.weights)

    # Static auxiliary data
    aux_data = {
        "variables": data_obj.variables,
        "has_weights": data_obj.weights is not None,
    }
    return children, aux_data


def _unbinned_data_unflatten(aux_data: dict, children: list) -> UnbinnedData:
    """Unflatten UnbinnedData from JAX PyTree."""
    data = children[0]
    weights = None
    if aux_data["has_weights"]:
        weights = children[1]

    return UnbinnedData(
        data=data,
        variables=aux_data["variables"],
        weights=weights,
    )


# Register UnbinnedData with JAX
jax.tree_util.register_pytree_node(
    UnbinnedData, _unbinned_data_flatten, _unbinned_data_unflatten
)
