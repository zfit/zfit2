from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

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


# JAX PyTree registration for Variable class
def _variable_flatten(var: Variable) -> Tuple[Tuple, Dict[str, Any]]:
    """Flatten a Variable for JAX PyTree."""
    # No dynamic values to track as children
    children = ()
    aux_data = {
        "name": var.name,
        "label": var.label,
        "lower": var.lower,
        "upper": var.upper,
    }
    return children, aux_data

def _variable_unflatten(aux_data: Dict[str, Any], children: Tuple) -> Variable:
    """Unflatten a Variable from JAX PyTree."""
    return Variable(
        name=aux_data["name"],
        label=aux_data["label"],
        lower=aux_data["lower"],
        upper=aux_data["upper"],
    )

# Register Variable class with JAX
jax.tree_util.register_pytree_node(
    Variable,
    _variable_flatten,
    _variable_unflatten
)
