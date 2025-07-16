from __future__ import annotations


class ZfitMinimizer:
    def minimize(self, loss, params=None, *, init=None):
        raise NotImplementedError


class ZfitStoppingCriterion:
    pass
