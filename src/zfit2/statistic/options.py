"""Options for statistical computations.

This module provides configuration options for statistical classes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class NLLOptions:
    """Options for NLL computation.

    Attributes:
        offset: The offset method to use. REQUIRED. Can be:
            - 'none' for no offset
            - 'mean:value' to subtract mean and start from value
            - 'median:value' to subtract median and start from value
            - 'elementwise:value' to subtract elementwise and start from value
            - A callable that takes logpdf values and returns offset values
        start_value: The value that NLL should start from (only used for string offsets)
    """

    offset: str | Callable[[Any], Any]
    start_value: float = 0.0

    def get_offset_method(self) -> str:
        """Get the offset method name.

        Returns:
            The offset method name
        """
        if callable(self.offset):
            return "custom"

        # Handle legacy format with colon
        if ":" in self.offset:
            method, value_str = self.offset.split(":", 1)
            # Override start_value if provided in string
            try:
                self.start_value = float(value_str)
            except ValueError:
                msg = f"Invalid start value in offset string: {value_str}"
                raise ValueError(msg)
            return method

        return self.offset

    @classmethod
    def none(cls, start_value: float = 0.0) -> NLLOptions:
        """Create options for no offset."""
        return cls(offset="none", start_value=start_value)

    @classmethod
    def mean(cls, start_value: float = 0.0) -> NLLOptions:
        """Create options for mean offset."""
        return cls(offset="mean", start_value=start_value)

    @classmethod
    def median(cls, start_value: float = 0.0) -> NLLOptions:
        """Create options for median offset."""
        return cls(offset="median", start_value=start_value)

    @classmethod
    def elementwise(cls, start_value: float = 0.0) -> NLLOptions:
        """Create options for elementwise offset."""
        return cls(offset="elementwise", start_value=start_value)

    @classmethod
    def custom(cls, offset_fn: Callable[[Any], Any]) -> NLLOptions:
        """Create options for custom offset function."""
        return cls(offset=offset_fn, start_value=0.0)
