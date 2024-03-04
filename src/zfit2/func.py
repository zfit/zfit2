from __future__ import annotations


class Variable:
    def __init__(self, name):
        self.name = name

    def as_vars(self):
        return Variables(self)


class Variables:
    def __init__(self, variables):
        if not isinstance(variables, list | tuple):
            variables = [variables]
        self.variables = variables

    @property
    def names(self):
        return [v.name for v in self.variables]

    def as_vars(self):
        return self


def convert_to_variables(var):
    if isinstance(var, list | tuple):
        return Variables([convert_to_variables(v) for v in var])
    raise NotImplementedError


class Func:
    def __init__(self, invars, outvars):
        self.input = convert_to_variables(invars)
        self.output = convert_to_variables(outvars)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
