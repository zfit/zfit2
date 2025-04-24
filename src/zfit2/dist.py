from __future__ import annotations


class Distribution:
    def __init__(self, domain, params, *, name=None, label=None):
        self.name = name
        self.label = label if label is not None else name
        self.params = params
        self.domain = domain

    def __repr__(self):
        return f"Distribution(name={self.name}, params={self.params})"

    def __str__(self):
        return f"Distribution {self.name}: {self.params}"

    def log_pdf(self, x, *, params=None, norm=None):
        """Logarithm of the probability density function."""
        msg = "log_pdf not implemented for this distribution"
        raise NotImplementedError(msg)

    def pdf(self, x, *, params=None, norm=None):
        """Probability density function."""
        msg = "pdf not implemented for this distribution"
        raise NotImplementedError(msg)

    def integrate(self, limits, *, params=None, norm=None):
        """Integrate the distribution over the given limits."""
        msg = "integrate not implemented for this distribution"
        raise NotImplementedError(msg)

    def sample(self, size, *, params=None, norm=None):
        """Sample from the distribution."""
        msg = "sample not implemented for this distribution"
        raise NotImplementedError(msg)
