from __future__ import annotations

from zfit2.variable import convert_to_variables


class Func:
    def __init__(self, domain, codomain):
        self.domain = convert_to_variables(domain)
        self.codomain = convert_to_variables(codomain)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
