from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

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

    def __float__(self):
        """Convert the variable to a float."""
        return float(self.value)

    def __array__(self):
        """Convert the variable to a numpy array."""
        return znp.array(self.value)


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

    def __len__(self):
        """Return the number of parameters."""
        return len(self.params)

    def values(self):
        """Return the values of the parameters."""
        return znp.array([param.value for param in self.params])

    def fixed(self):
        """Return the fixed status of the parameters."""
        return znp.array([param.fixed for param in self.params])

    def stepsizes(self):
        """Return the stepsizes of the parameters."""
        # Convert to numpy array with float64 dtype to match test expectations
        return znp.array([param.stepsize for param in self.params], dtype=jnp.float64)

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


# JAX PyTree registration for Parameter class
def _parameter_flatten(param: Parameter) -> tuple[tuple[float], dict[str, Any]]:
    """Flatten a Parameter for JAX PyTree."""
    # The value is dynamic
    children = (param.value,)
    aux_data = {
        "name": param.name,
        "label": param.label,
        "lower": param.lower,
        "upper": param.upper,
        "stepsize": param.stepsize,
        "prior": param.prior,
        "fixed": param.fixed,
    }
    return children, aux_data


def _parameter_unflatten(aux_data: dict[str, Any], children: tuple[float]) -> Parameter:
    """Unflatten a Parameter from JAX PyTree."""
    (value,) = children
    return Parameter(
        name=aux_data["name"],
        value=value,
        lower=aux_data["lower"],
        upper=aux_data["upper"],
        stepsize=aux_data["stepsize"],
        prior=aux_data["prior"],
        fixed=aux_data["fixed"],
        label=aux_data["label"],
    )


# Register Parameter class with JAX
jax.tree_util.register_pytree_node(Parameter, _parameter_flatten, _parameter_unflatten)
