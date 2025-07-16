"""Options for statistical computations.

This module provides configuration options for statistical classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T", bound="NLLOptions")


class NLLOptions:
    """Universal options for NLL computations with chainable interface.

    This class provides a fluent interface for configuring NLL computations.
    Each method returns a new instance with the updated configuration.

    Examples:
        >>> options = NLLOptions().offset("mean", start_value=10000)
        >>> options = NLLOptions().offset("median", start_value=5000).sum("kahan")
        >>> options = NLLOptions().offset(custom_fn)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize options with given configuration.

        Args:
            config: Initial configuration dictionary
        """
        self._config: dict[str, Any] = config or {}
        self._configured_methods: set[str] = set()

        # Convert numerical values to JAX arrays
        self._ensure_jax_arrays()

    def _ensure_jax_arrays(self) -> None:
        """Ensure all numerical values in config are JAX arrays."""
        if (
            offset_config := self._config.get("offset")
        ) and "start_value" in offset_config:
            offset_config["start_value"] = jnp.asarray(offset_config["start_value"])

    def offset(
        self: T,
        method: str | Callable[[jnp.ndarray], jnp.ndarray],
        *,
        start_value: float = 0.0,
        force: bool = False,
    ) -> T:
        """Configure offset method.

        Args:
            method: Offset method. Can be:
                - 'none': No offset
                - 'mean': Subtract mean
                - 'median': Subtract median
                - 'elementwise': Subtract elementwise
                - A callable taking logpdf values and returning offset values
            start_value: Target value after offset (only for string methods)
            force: If True, allow reconfiguring even if already set

        Returns:
            New options instance with offset configured

        Raises:
            ValueError: If offset is already configured and force=False
        """
        if "offset" in self._configured_methods and not force:
            msg = (
                "Offset method already configured. Use force=True to override or "
                "create a new options instance."
            )
            raise ValueError(msg)

        new_config = self._config.copy()
        new_configured = self._configured_methods.copy()

        if callable(method):
            new_config["offset"] = {"method": "custom", "function": method}
        else:
            valid_methods = {"none", "mean", "median", "elementwise"}
            if method not in valid_methods:
                msg = (
                    f"Invalid offset method: {method}. "
                    f"Valid methods are: {', '.join(sorted(valid_methods))}"
                )
                raise ValueError(msg)
            new_config["offset"] = {
                "method": method,
                "start_value": jnp.asarray(start_value),
            }

        new_configured.add("offset")

        new_instance = self.__class__(new_config)
        new_instance._configured_methods = new_configured
        return new_instance

    def sum(self: T, method: str = "standard", *, force: bool = False) -> T:
        """Configure summation method.

        Args:
            method: Summation method. Currently supported:
                - 'standard': Standard summation (default)
                Future options might include 'kahan', 'pairwise', etc.
            force: If True, allow reconfiguring even if already set

        Returns:
            New options instance with summation configured

        Raises:
            ValueError: If sum is already configured and force=False
        """
        if "sum" in self._configured_methods and not force:
            msg = (
                "Sum method already configured. Use force=True to override or "
                "create a new options instance."
            )
            raise ValueError(msg)

        new_config = self._config.copy()
        new_configured = self._configured_methods.copy()

        valid_methods = {"standard"}  # Can be extended in future
        if method not in valid_methods:
            msg = (
                f"Invalid sum method: {method}. "
                f"Valid methods are: {', '.join(sorted(valid_methods))}"
            )
            raise ValueError(msg)

        new_config["sum"] = {"method": method}
        new_configured.add("sum")

        new_instance = self.__class__(new_config)
        new_instance._configured_methods = new_configured
        return new_instance

    def get_offset_config(self) -> dict[str, Any] | None:
        """Get offset configuration.

        Returns:
            Offset configuration dict or None if not configured
        """
        return self._config.get("offset")

    def get_sum_config(self) -> dict[str, Any] | None:
        """Get sum configuration.

        Returns:
            Sum configuration dict or None if not configured
        """
        return self._config.get("sum")

    def __repr__(self) -> str:
        """String representation of options."""
        parts = []

        if offset := self.get_offset_config():
            if offset["method"] == "custom":
                parts.append("offset=custom")
            else:
                parts.append(f"offset={offset['method']}")
                if "start_value" in offset:
                    value = offset["start_value"]
                    # Convert JAX array to Python float for display
                    if hasattr(value, "item"):
                        value = value.item()
                    # Format as int if it's a whole number
                    if isinstance(value, float) and value.is_integer():
                        parts.append(f"start_value={int(value)}")
                    else:
                        parts.append(f"start_value={value}")

        if sum_cfg := self.get_sum_config():
            parts.append(f"sum={sum_cfg['method']}")

        return f"NLLOptions({', '.join(parts)})"

    # Convenience factory methods for common configurations
    @classmethod
    def default(cls) -> NLLOptions:
        """Create default options (mean offset with start_value=10000)."""
        return cls().offset("mean", start_value=10000.0)

    @classmethod
    def none(cls, start_value: float = 0.0) -> NLLOptions:
        """Create options with no offset."""
        return cls().offset("none", start_value=start_value)

    @classmethod
    def mean(cls, start_value: float = 10000.0) -> NLLOptions:
        """Create options with mean offset."""
        return cls().offset("mean", start_value=start_value)

    @classmethod
    def median(cls, start_value: float = 10000.0) -> NLLOptions:
        """Create options with median offset."""
        return cls().offset("median", start_value=start_value)

    @classmethod
    def elementwise(cls, start_value: float = 10000.0) -> NLLOptions:
        """Create options with elementwise offset."""
        return cls().offset("elementwise", start_value=start_value)

    @classmethod
    def custom(cls, offset_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> NLLOptions:
        """Create options with custom offset function."""
        return cls().offset(offset_fn)

    @property
    def start_value(self) -> float:
        """Get start value for backwards compatibility."""
        config = self.get_offset_config()
        if config and "start_value" in config:
            value = config["start_value"]
            # Convert JAX array to Python float
            if hasattr(value, "item"):
                return value.item()
            return value
        return 0.0

    def get_offset_method(self) -> str:
        """Get the offset method name.

        Returns:
            The offset method name
        """
        if config := self.get_offset_config():
            return config["method"]
        return "none"


# JAX PyTree registration for NLLOptions
def _nll_options_flatten(options: NLLOptions) -> tuple[list[Any], dict[str, Any]]:
    """Flatten NLLOptions for JAX PyTree."""
    # Extract dynamic values (numerical values and functions)
    children = []
    aux_data = {"methods": {}}

    if offset_config := options._config.get("offset"):
        if "start_value" in offset_config:
            children.append(offset_config["start_value"])
        else:
            children.append(None)

        if offset_config["method"] == "custom":
            # Custom functions are static data
            aux_data["methods"]["offset"] = {
                "method": "custom",
                "function": offset_config.get("function"),
            }
        else:
            aux_data["methods"]["offset"] = {"method": offset_config["method"]}
    else:
        children.append(None)

    if sum_config := options._config.get("sum"):
        aux_data["methods"]["sum"] = sum_config

    aux_data["configured_methods"] = options._configured_methods

    return children, aux_data


def _nll_options_unflatten(aux_data: dict[str, Any], children: list[Any]) -> NLLOptions:
    """Unflatten NLLOptions from JAX PyTree."""
    config = {}

    # Reconstruct offset config
    start_value = children[0]
    if "offset" in aux_data["methods"]:
        offset_info = aux_data["methods"]["offset"]
        config["offset"] = {"method": offset_info["method"]}

        if start_value is not None:
            config["offset"]["start_value"] = start_value

        if "function" in offset_info:
            config["offset"]["function"] = offset_info["function"]

    # Reconstruct sum config
    if "sum" in aux_data["methods"]:
        config["sum"] = aux_data["methods"]["sum"]

    # Create new instance
    options = NLLOptions(config)
    options._configured_methods = aux_data["configured_methods"]

    return options


# Register NLLOptions with JAX
jax.tree_util.register_pytree_node(
    NLLOptions, _nll_options_flatten, _nll_options_unflatten
)


# Legacy compatibility class that creates an NLLOptions instance
class NLLOptionsLegacy:
    """Legacy NLLOptions interface for backwards compatibility.

    This class maintains the old constructor interface but internally
    uses the new NLLOptions implementation.
    """

    def __new__(
        cls, offset: str | Callable[[Any], Any], start_value: float = 0.0
    ) -> NLLOptions:
        """Create NLLOptions using legacy interface.

        Args:
            offset: Offset method or callable
            start_value: Start value for offset

        Returns:
            NLLOptions instance
        """
        # Handle legacy colon format
        if isinstance(offset, str) and ":" in offset:
            method, value_str = offset.split(":", 1)
            try:
                start_value = float(value_str)
            except ValueError:
                msg = f"Invalid start value in offset string: {value_str}"
                raise ValueError(msg)
            offset = method

        # Create new NLLOptions
        if callable(offset):
            return NLLOptions().offset(offset)
        else:
            return NLLOptions().offset(offset, start_value=start_value)
