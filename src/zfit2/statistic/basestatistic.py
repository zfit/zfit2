"""Base class for statistical tests and test statistics in zfit2.

This module provides the foundation for implementing various statistical tests
such as likelihood ratio tests, chi-square tests, and other hypothesis tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStatistic(ABC):
    """Abstract base class for test statistics.

    This class provides the foundation for implementing various test statistics
    used in hypothesis testing, goodness-of-fit tests, and model comparison.

    Attributes:
        name: Machine-readable identifier for the statistic (e.g., 'chi2', 'lr_test')
        label: Human-readable label for display (e.g., 'Chi-square test')
    """

    def __init__(self, *, name: str, label: str | None = None):
        """Initialize the base statistic.

        Args:
            name: Machine-readable identifier for the statistic
            label: Human-readable label. If None, uses the name.
        """
        self.name = name
        self.label = label if label is not None else name

    def __repr__(self) -> str:
        """Return a string representation of the statistic."""
        return f"{self.__class__.__name__}(name='{self.name}', label='{self.label}')"

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        return f"{self.label}"

    @abstractmethod
    def value(self, *args, **kwargs) -> Any:
        """Compute the test statistic value.

        This method must be implemented by subclasses to compute the actual
        test statistic value based on the provided data or model parameters.

        Returns:
            The computed test statistic value
        """
        msg = "Subclasses must implement value()"
        raise NotImplementedError(msg)

    def __call__(self, *args, **kwargs) -> Any:
        """Compute the test statistic by calling the value method.

        Returns:
            The computed test statistic value
        """
        return self.value(*args, **kwargs)


# JAX PyTree registration for BaseStatistic
def _base_statistic_flatten(stat: BaseStatistic) -> tuple[tuple, dict]:
    """Flatten BaseStatistic for JAX PyTree."""
    # No dynamic values
    children = ()
    aux_data = {
        "name": stat.name,
        "label": stat.label,
        "class": stat.__class__,
    }
    return children, aux_data


def _base_statistic_unflatten(aux_data: dict, children: tuple) -> BaseStatistic:
    """Unflatten BaseStatistic from JAX PyTree."""
    cls = aux_data["class"]
    return cls(name=aux_data["name"], label=aux_data["label"])


# Note: We don't register BaseStatistic itself since it's abstract,
# but subclasses can use this pattern for their own registration
