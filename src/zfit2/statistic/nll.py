"""Negative log-likelihood and related statistics.

This module provides classes for computing negative log-likelihood
and related statistics for parameter estimation and model fitting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from ..util import to_collection
from .basestatistic import BaseStatistic
from .options import NLLOptions


class BaseNLL(BaseStatistic, ABC):
    """Base class for negative log-likelihood-like statistics.

    This class provides a common interface for NLL-based statistics,
    with preprocessing of inputs and private methods for the actual computation.
    """

    def value(self, *args, full: bool = False, **kwargs) -> float | jnp.ndarray:
        """Compute the statistic value with preprocessing.

        This method preprocesses inputs before calling the private
        _value method for the actual computation.

        Args:
            full: If True, compute the full value without any offset or approximation.
                  If False (default), apply offsets and approximations as configured.

        Returns:
            The computed statistic value
        """
        # Call precompile if the subclass has it
        if (
            hasattr(self, "precompile")
            and hasattr(self, "_precompiled")
            and not self._precompiled
            and not full
        ):
            self.precompile(*args, **kwargs)

        return self._value(*args, full=full, **kwargs)

    @abstractmethod
    def _value(self, *args, full: bool = False, **kwargs) -> float | jnp.ndarray:
        """Private method for computing the actual statistic value.

        This method must be implemented by subclasses.

        Args:
            full: If True, compute full value without offset

        Returns:
            The computed statistic value
        """
        msg = "Subclasses must implement _value()"
        raise NotImplementedError(msg)

    def loglike(self, params: dict | None = None) -> jnp.ndarray:
        """Compute log probability density values with preprocessing.

        Args:
            params: Optional parameter values to use for evaluation

        Returns:
            Array of log probability density values for each data point
        """
        # Preprocessing would go here
        return self._loglike(params)

    @abstractmethod
    def _loglike(self, params: dict | None = None) -> jnp.ndarray:
        """Private method for computing log probability density values.

        Args:
            params: Optional parameter values to use for evaluation

        Returns:
            Array of log probability density values for each data point
        """
        msg = "Subclasses must implement _loglike()"
        raise NotImplementedError(msg)

    def sum(self, ll: jnp.ndarray) -> jnp.ndarray:
        """Sum the log-likelihood values with preprocessing.

        Args:
            ll: Array of log-likelihood values

        Returns:
            Sum of log-likelihood values
        """
        # Preprocessing would go here
        return self._sum(ll)

    @abstractmethod
    def _sum(self, ll: jnp.ndarray) -> jnp.ndarray:
        """Private method for summing log-likelihood values.

        Args:
            ll: Array of log-likelihood values

        Returns:
            Sum of log-likelihood values
        """
        msg = "Subclasses must implement _sum()"
        raise NotImplementedError(msg)


class NLL(BaseNLL):
    """Negative log-likelihood statistic.

    This class computes the negative log-likelihood for given distributions
    and data, used for parameter estimation via maximum likelihood.

    Attributes:
        dists: List of distributions to evaluate
        data: The observed data
    """

    def __init__(
        self,
        dists: Any,
        data: Any,
        options: NLLOptions | None = None,
        *,
        name: str = "nll",
        label: str | None = None,
    ):
        """Initialize the NLL statistic.

        Args:
            dists: Distribution or collection of distributions
            data: Observed data or collection of data (must have exactly one dataset per distribution)
            options: Configuration options for NLL computation (defaults to mean offset with start_value=10000)
            name: Machine-readable identifier (default: "nll")
            label: Human-readable label (default: "Negative Log-Likelihood")
        """
        if options is None:
            options = NLLOptions.default()
        if label is None:
            label = "Negative Log-Likelihood"

        super().__init__(name=name, label=label)

        # Convert to collections (as lists) for consistency
        dists = to_collection(dists, collection_type=list, force=True)
        data = to_collection(data, collection_type=list, force=True)

        # Check lengths match - no broadcasting allowed
        if len(dists) != len(data):
            msg = (
                f"Number of distributions ({len(dists)}) must exactly match "
                f"number of datasets ({len(data)}). No broadcasting is allowed."
            )
            raise ValueError(msg)

        self.dists = dists
        self.data = data
        self.options = options

        # Precompilation state
        self._precompiled = False
        self._offset_values = None
        self._adjustment = None

    def _loglike(self, params: dict | None = None) -> jnp.ndarray:
        """Private method for computing log probability density values.

        Args:
            params: Optional parameter values to use for evaluation

        Returns:
            Array of log probability density values for each data point
        """
        if len(self.dists) == 0:
            return jnp.array([])

        logpdfs = []
        for dist, data in zip(self.dists, self.data, strict=True):
            # Call logpdf method on distribution
            if hasattr(dist, "logpdf"):
                # If distribution has params argument
                try:
                    logpdf = dist.logpdf(data, params=params)
                except TypeError:
                    # Try without params argument (e.g., scipy distributions)
                    logpdf = dist.logpdf(data)
            else:
                msg = f"Distribution {dist} does not have a logpdf method"
                raise AttributeError(msg)

            logpdfs.append(jnp.asarray(logpdf))

        # Return log pdfs based on the case
        if len(logpdfs) == 1:
            return logpdfs[0]
        else:
            # Check if all arrays have the same shape
            shapes = [logpdf.shape for logpdf in logpdfs]
            if all(shape == shapes[0] for shape in shapes):
                # Same shape - assume product distribution (sum log pdfs element-wise)
                return jnp.sum(jnp.stack(logpdfs), axis=0)
            else:
                # Different shapes - concatenate all log pdfs
                return jnp.concatenate(logpdfs)

    def _sum(self, ll: jnp.ndarray) -> jnp.ndarray:
        """Private method for summing log-likelihood values.

        Args:
            ll: Array of log-likelihood values

        Returns:
            Sum of log-likelihood values
        """
        return jnp.sum(ll)

    def precompile(self, params: dict | None = None) -> None:
        """Precompile offset values based on initial logpdf evaluation.

        This method computes the offset values and adjustment factor
        to ensure the NLL starts at the specified value.

        Args:
            params: Optional parameter values for initial evaluation
        """
        if self._precompiled:
            return

        # Get initial logpdf values
        loglike_values = self._loglike(params)

        # Get offset configuration
        offset_config = self.options.get_offset_config()
        if offset_config is None:
            # No offset configured, use default
            offset_config = {"method": "mean", "start_value": 10000.0}

        method = offset_config["method"]

        # Calculate offset based on method
        if method == "none":
            self._offset_values = jnp.zeros_like(loglike_values)
            # For 'none', we don't adjust - just return raw NLL
            self._adjustment = 0.0
        elif method == "mean":
            offset = jnp.mean(loglike_values)
            self._offset_values = jnp.full_like(loglike_values, offset)
            # Calculate adjustment
            adjusted_loglike = loglike_values - self._offset_values
            current_nll = -jnp.sum(adjusted_loglike)
            self._adjustment = offset_config.get("start_value", 0.0) - current_nll
        elif method == "median":
            offset = jnp.median(loglike_values)
            self._offset_values = jnp.full_like(loglike_values, offset)
            # Calculate adjustment
            adjusted_loglike = loglike_values - self._offset_values
            current_nll = -jnp.sum(adjusted_loglike)
            self._adjustment = offset_config.get("start_value", 0.0) - current_nll
        elif method == "elementwise":
            self._offset_values = loglike_values
            # Calculate adjustment
            adjusted_loglike = loglike_values - self._offset_values
            current_nll = -jnp.sum(adjusted_loglike)
            self._adjustment = offset_config.get("start_value", 0.0) - current_nll
        elif method == "custom":
            # Custom offset function
            custom_fn = offset_config.get("function")
            if custom_fn is None:
                msg = "Custom offset method configured but no function provided"
                raise ValueError(msg)
            self._offset_values = custom_fn(loglike_values)
            # For custom functions, we don't adjust
            self._adjustment = 0.0
        else:
            msg = (
                f"Unknown offset method: {method}. "
                f"Valid methods are: 'none', 'mean', 'median', 'elementwise', or a callable."
            )
            raise ValueError(msg)

        self._precompiled = True

    def _value(self, params: dict | None = None, full: bool = False) -> jnp.ndarray:
        """Compute the negative log-likelihood.

        Args:
            params: Optional parameter values
            full: If True, compute full value without offset

        Returns:
            The negative log-likelihood value
        """
        # Compute log-likelihood values
        loglike_values = self._loglike(params)

        if full:
            # No offset, just return negative sum
            summed = self._sum(loglike_values)
            return -summed
        else:
            # Precompile on first call
            if not self._precompiled:
                self.precompile(params)

            # Apply offset
            adjusted_loglike = loglike_values - self._offset_values

            # Sum them
            summed = self._sum(adjusted_loglike)

            # Return negative plus adjustment
            return -summed + self._adjustment


# JAX PyTree registration for NLL
def _nll_flatten(nll: NLL) -> tuple[list, dict]:
    """Flatten NLL for JAX PyTree."""
    # Dynamic values (distributions and data might contain parameters)
    # Include offset values and adjustment if precompiled
    children = (
        [nll.dists, nll.data, nll._offset_values, nll._adjustment]
        if nll._precompiled
        else [nll.dists, nll.data]
    )

    # Static auxiliary data
    aux_data = {
        "name": nll.name,
        "label": nll.label,
        "options": nll.options,
        "precompiled": nll._precompiled,
    }
    return children, aux_data


def _nll_unflatten(aux_data: dict, children: list) -> NLL:
    """Unflatten NLL from JAX PyTree."""
    if aux_data["precompiled"]:
        dists, data, offset_values, adjustment = children
        nll = NLL(
            dists=dists,
            data=data,
            options=aux_data["options"],
            name=aux_data["name"],
            label=aux_data["label"],
        )
        nll._precompiled = True
        nll._offset_values = offset_values
        nll._adjustment = adjustment
    else:
        dists, data = children
        nll = NLL(
            dists=dists,
            data=data,
            options=aux_data["options"],
            name=aux_data["name"],
            label=aux_data["label"],
        )
    return nll


# Register NLL with JAX
jax.tree_util.register_pytree_node(NLL, _nll_flatten, _nll_unflatten)
