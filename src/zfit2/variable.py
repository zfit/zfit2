from __future__ import annotations

from typing import Any

import jax

from .backend import numpy as znp


class Variable:
    def __init__(self, name, *, label=None, lower=None, upper=None):
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
    from .parameter import Parameter, Parameters

    if isinstance(var, Variable | Parameter):
        return Variables(var)
    if isinstance(var, Variables):
        return var  # Already a Variables object, return as is
    if isinstance(var, Parameters):
        return Variables([v for v in var.params])  # Extract Parameter objects
    if isinstance(var, list | tuple):
        # Flatten the list and extract individual Variable/Parameter objects
        flat_vars = []
        for v in var:
            if isinstance(v, Variable | Parameter):
                flat_vars.append(v)
            elif isinstance(v, Variables):
                flat_vars.extend(v.variables)
            elif isinstance(v, Parameters):
                flat_vars.extend(v.params)
            elif isinstance(v, list | tuple):
                # Recursively convert nested lists/tuples
                vars_obj = convert_to_variables(v)
                flat_vars.extend(vars_obj.variables)
            else:
                raise TypeError(f"Cannot convert {type(v)} to Variables")
        return Variables(flat_vars)
    raise NotImplementedError(f"Cannot convert {type(var)} to Variables")


# JAX PyTree registration for Variable class
def _variable_flatten(var: Variable) -> tuple[tuple, dict[str, Any]]:
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


def _variable_unflatten(aux_data: dict[str, Any], children: tuple) -> Variable:
    """Unflatten a Variable from JAX PyTree."""
    return Variable(
        name=aux_data["name"],
        label=aux_data["label"],
        lower=aux_data["lower"],
        upper=aux_data["upper"],
    )


# Register Variable class with JAX
jax.tree_util.register_pytree_node(Variable, _variable_flatten, _variable_unflatten)
