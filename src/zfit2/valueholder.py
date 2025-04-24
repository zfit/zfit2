from __future__ import annotations

from typing import Union

import numpy as np


class ValueHolder:
    """A dict-like class for storing numerical values (floats and arrays)."""

    def __init__(self, values: dict[str, Union[float, np.ndarray]] | None = None):
        self._values = {} if values is None else dict(values)

    def __getitem__(self, key: str) -> Union[float, np.ndarray]:
        return self._values[key]

    def __setitem__(self, key: str, value: Union[float, np.ndarray]) -> None:
        if not isinstance(value, float | np.ndarray):
            msg = f"Value must be float or numpy.ndarray, not {type(value)}"
            raise TypeError(msg)
        self._values[key] = value

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __add__(self, other: ValueHolder) -> ValueHolder:
        """Merge two Values objects, with other taking precedence for overlapping keys."""
        if not isinstance(other, ValueHolder):
            msg = f"Can only add Values objects, not {type(other)}"
            raise TypeError(msg)
        merged = ValueHolder(self._values.copy())
        merged._values.update(other._values)
        return merged
