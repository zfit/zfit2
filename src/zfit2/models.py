from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from zfit2.backend import numpy as znp
from zfit2.dist import Distribution
from zfit2.func import Func, AnalyticFunc
from zfit2.parameter import Parameter, Parameters
from zfit2.variable import Variable, Variables, convert_to_variables
from zfit2.integration import integrate


class Model(Distribution):
    """A statistical model based on a function.

    A Model is a Distribution that is defined by a function.
    It can be used for fitting and statistical analysis.
    """

    def __init__(
            self,
            func: Func,
            domain: Union[Variable, Variables, Sequence[Variable]],
            norm_range: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
            name: Optional[str] = None,
            label: Optional[str] = None
    ):
        """Initialize a model.

        Args:
            func: The function that defines the model
            domain: Domain of the model
            norm_range: Normalization range
            name: Name of the model
            label: Label of the model
        """
        self.func = func
        self.norm_range = norm_range or self._default_norm_range(domain)

        super().__init__(
            domain=domain,
            params=func.parameters,
            name=name or f"Model_{func.name}",
            label=label or f"Model of {func.name}"
        )

        # Cache for normalization factor
        self._norm_factor = None

    def _default_norm_range(self, domain: Union[Variable, Variables, Sequence[Variable]]) -> List[Tuple[float, float]]:
        """Get default normalization range from domain."""
        domain_vars = convert_to_variables(domain)
        return [(var.lower, var.upper) for var in domain_vars.variables]

    def _normalize(self, x, params=None, norm=None):
        """Normalize the function value."""
        # Get normalization factor
        if norm is None:
            norm = self.norm_range

        norm_factor = self._get_normalization_factor(params, norm)

        # Evaluate function
        func_values = self.func(x, parameters=params)

        # Normalize
        return func_values / norm_factor

    def _get_normalization_factor(self, params=None, norm=None):
        """Get the normalization factor for the model."""
        if self._norm_factor is not None and params is None and norm is None:
            return self._norm_factor

        # Calculate integral over normalization range
        norm_factor = integrate(self.func, norm or self.norm_range, parameters=params)

        # Cache result if using default parameters and range
        if params is None and norm is None:
            self._norm_factor = norm_factor

        return norm_factor

    def log_pdf(self, x, *, params=None, norm=None):
        """Logarithm of the probability density function."""
        pdf_val = self._normalize(x, params=params, norm=norm)
        return jnp.log(pdf_val)

    def pdf(self, x, *, params=None, norm=None):
        """Probability density function."""
        return self._normalize(x, params=params, norm=norm)

    def integrate(self, limits, *, params=None, norm=None):
        """Integrate the model over the given limits."""
        if norm is None:
            norm = self.norm_range

        # Get normalization factor
        norm_factor = self._get_normalization_factor(params, norm)

        # Calculate integral over the given limits
        integral = integrate(self.func, limits, parameters=params)

        # Normalize
        return integral / norm_factor

    def sample(self, size, *, params=None, norm=None):
        """Sample from the model."""
        # For now, use rejection sampling
        from zfit2.sampling import rejection_sample
        return rejection_sample(self, size, params=params, norm=norm)


class UnbinnedModel(Model):
    """Model for unbinned fitting."""

    def fit(self, data, minimizer=None, **minimizer_kwargs):
        """Fit the model to data."""
        from zfit2.likelihood import NLL
        from zfit2.minimize import minimize

        # Create negative log-likelihood
        nll = NLL(distribution=self, data=data)

        # Minimize
        return minimize(nll, minimizer=minimizer, **minimizer_kwargs)


class BinnedModel(Model):
    """Model for binned fitting."""

    def fit(self, hist, minimizer=None, **minimizer_kwargs):
        """Fit the model to a histogram."""
        from zfit2.likelihood import BinnedNLL
        from zfit2.minimize import minimize

        # Create binned negative log-likelihood
        nll = BinnedNLL(distribution=self, hist=hist)

        # Minimize
        return minimize(nll, minimizer=minimizer, **minimizer_kwargs)


class GaussianModel(Model):
    """Gaussian model: A * exp(-0.5 * ((x - mu) / sigma)^2)."""

    def __init__(
            self,
            domain: Union[Variable, Variables, Sequence[Variable]],
            mu: Parameter,
            sigma: Parameter,
            amplitude: Optional[Parameter] = None,
            name: Optional[str] = None,
            label: Optional[str] = None
    ):
        """Initialize a Gaussian model.

        Args:
            domain: Domain of the model
            mu: Mean parameter
            sigma: Standard deviation parameter
            amplitude: Amplitude parameter
            name: Name of the model
            label: Label of the model
        """
        from zfit2.func import GaussianFunc

        # Create Gaussian function
        func = GaussianFunc(domain=domain, mu=mu, sigma=sigma, amplitude=amplitude, name=f"GaussianFunc_{name}")

        super().__init__(
            func=func,
            domain=domain,
            name=name or "GaussianModel",
            label=label or "Gaussian Model"
        )


class PolynomialModel(Model):
    """Polynomial model: sum(a_i * x^i)."""

    def __init__(
            self,
            domain: Union[Variable, Variables, Sequence[Variable]],
            coefficients: Union[Parameter, Parameters, Sequence[Parameter]],
            degree: Optional[int] = None,
            name: Optional[str] = None,
            label: Optional[str] = None
    ):
        """Initialize a polynomial model.

        Args:
            domain: Domain of the model
            coefficients: Coefficients of the polynomial
            degree: Degree of the polynomial (inferred from coefficients if None)
            name: Name of the model
            label: Label of the model
        """
        from zfit2.func import PolynomialFunc

        # Create polynomial function
        func = PolynomialFunc(
            domain=domain,
            coefficients=coefficients,
            degree=degree,
            name=f"PolyFunc_{name}"
        )

        super().__init__(
            func=func,
            domain=domain,
            name=name or "PolynomialModel",
            label=label or "Polynomial Model"
        )