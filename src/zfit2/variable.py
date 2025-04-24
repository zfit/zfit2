from __future__ import annotations

from .backend import numpy as znp


class Variable:
    def __init__(self, name,*, label=None, lower=None, upper=None):
        self.name = name
        self.label = label if label is not None else name
        self.lower = lower
        self.upper = upper

    def to_vars(self):
        return Variables(self)


class Variables:
    def __init__(self, variables):
        if not isinstance(variables, list | tuple | Variables):
            variables = [variables]
        self.variables = list(variables)

    @property
    def names(self):
        return [v.name for v in self.variables]

    @property
    def labels(self):
        return [v.label for v in self.variables]

    @property
    def lowers(self):
        return znp.asarray([v.lower for v in self.variables])

    @property
    def uppers(self):
        return znp.asarray([v.upper for v in self.variables])

    def to_vars(self):
        return self

    def __getitem__(self, key):
        """Return Variable by index or name."""
        if isinstance(key, str):
            for var in self.variables:
                if var.name == key:
                    return var
            msg = f"Variable with name {key} not found."
            raise KeyError(msg)
        return self.variables[key]


def convert_to_variables(var):
    if isinstance(var, list | tuple):
        return Variables([convert_to_variables(v) for v in var])
    raise NotImplementedError
