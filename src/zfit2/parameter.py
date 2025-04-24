from __future__ import annotations

from .backend import numpy as znp
from .variable import Variable, Variables


class Parameter(Variable):
    def __init__(
        self,
        name,
        value,
        lower,
        upper,
        *,
        stepsize=None,
        prior=None,
        fixed=False,
        label=None,
    ):
        super().__init__(name=name, label=label, lower=lower, upper=upper)
        self.value = value
        self.stepsize = stepsize
        self.prior = prior
        self.fixed = fixed
        self.label = label if label is not None else name

    def __repr__(self):
        return f"Parameter(name={self.name}, value={self.value}, lower={self.lower}, upper={self.upper}, stepsize={self.stepsize})"

    def __str__(self):
        return f"Parameter {self.name}: {self.value} (fixed={self.fixed})"

    def to_params(self):
        """Convert to a list of parameters."""
        return Parameters(self)


class Parameters(Variables):
    def __init__(self, params):
        """A class to hold a collection of parameters."""
        if not isinstance(params, list | tuple | Parameters):
            params = [params]
        if not all(isinstance(p, Parameter) for p in params):
            msg = "All elements must be of type Parameter"
            raise TypeError(msg)
        super().__init__(params)

    @property
    def params(self):
        """Return the list of parameters."""
        return self.variables

    def values(self):
        """Return the values of the parameters."""
        return znp.array([param.value for param in self.params])

    def fixed(self):
        """Return the fixed status of the parameters."""
        return znp.array([param.fixed for param in self.params])

    def stepsizes(self):
        """Return the stepsizes of the parameters."""
        return znp.array([param.stepsize for param in self.params])

    def __add__(self, other):
        """Add two Parameters or Parameters objects."""
        if isinstance(other, Parameter):
            return Parameters([*self.params, other])
        elif isinstance(other, Parameters):
            return Parameters(self.params + other.params)
        else:
            msg = "Can only add Parameter or Parameters objects"
            raise TypeError(msg)

    def to_params(self):
        """Convert to a list of parameters."""
        return Parameters(self)

    def __array__(self):
        """Return the values of the parameters as a numpy array."""
        return self.values()
