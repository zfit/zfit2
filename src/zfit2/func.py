from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp

from zfit2.parameter import Parameter, Parameters
from zfit2.variable import Variable, Variables, convert_to_variables


class FuncProperties:
    """Properties of a function that help with integration and optimization."""

    def __init__(self):
        self.is_linear = False
        self.is_monotonic = False
        self.is_positive = False
        self.has_analytic_integral = False
        self.supported_domains = []
        self.linear_parameters = set()


class Func:
    """Base class for all function objects in zfit2.

    A Func maps from a domain to a codomain with a specific functional form.
    Functions can be composed, transformed, and parameterized.
    """

    def __init__(
        self,
        domain: Union[Variable, Variables, Sequence[Variable]],
        codomain: Union[Variable, Variables, Sequence[Variable]],
        parameters: Optional[Union[Parameter, Parameters, Sequence[Parameter]]] = None,
        name: Optional[str] = None,
    ):
        """Initialize a function.

        Args:
            domain: Input space of the function
            codomain: Output space of the function
            parameters: Parameters of the function
            name: Name of the function
        """
        self.domain = convert_to_variables(domain)
        self.codomain = convert_to_variables(codomain)

        # Convert parameters to Parameters object
        if parameters is None:
            self.parameters = Parameters([])
        elif isinstance(parameters, Parameter):
            self.parameters = Parameters([parameters])
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            self.parameters = Parameters(list(parameters))

        self.name = name or self.__class__.__name__

        # Initialize properties for integration and optimization
        self.properties = FuncProperties()

        # Initialize cache for JIT-compiled versions
        self._jitted_func = None
        self._jitted_grad = None

    def __call__(self, *args, **kwargs) -> Any:
        """Apply the function to inputs."""
        parameters = kwargs.pop("parameters", None)
        return self._call_impl(args, parameters=parameters, **kwargs)

    def _call_impl(self, x, parameters=None, **kwargs) -> Any:
        """Implementation of the function call."""
        raise NotImplementedError("Subclasses must implement _call_impl")

    def is_linear_in(self, parameter: Union[str, Parameter]) -> bool:
        """Check if the function is linear in a parameter."""
        if isinstance(parameter, Parameter):
            parameter = parameter.name
        return parameter in self.properties.linear_parameters

    def compose(self, other: Func) -> ComposedFunc:
        """Compose this function with another function: f(g(x))."""
        return ComposedFunc(self, other)

    def with_parameters(self, parameters: dict[str, Any]) -> Func:
        """Create a new function with updated parameter values."""
        new_func = self.__class__(
            domain=self.domain,
            codomain=self.codomain,
            parameters=self.parameters,
            name=self.name,
        )

        # Update parameter values
        for param in new_func.parameters.params:
            if param.name in parameters:
                param.value = parameters[param.name]

        return new_func

    def jit(self) -> Callable:
        """Return a JIT-compiled version of the function."""
        if self._jitted_func is None:
            self._jitted_func = jax.jit(self.__call__)
        return self._jitted_func

    def grad(self, parameters: Optional[list[str]] = None) -> Callable:
        """Return a function that computes the gradient w.r.t. parameters."""
        if parameters is None:
            parameters = [param.name for param in self.parameters.params]

        def grad_func(*args, **kwargs):
            # Define a wrapper function that extracts parameters
            def wrapped(param_values):
                param_dict = {
                    name: value
                    for name, value in zip(parameters, param_values, strict=False)
                }
                return self(*args, parameters=param_dict, **kwargs)

            # Get current parameter values
            param_values = [self.parameters[name].value for name in parameters]

            # Calculate gradient
            return jax.grad(wrapped)(param_values)

        return grad_func


class AnalyticFunc(Func):
    """A function defined by an analytic expression."""

    def __init__(
        self,
        domain: Union[Variable, Variables, Sequence[Variable]],
        codomain: Union[Variable, Variables, Sequence[Variable]],
        func: Callable,
        parameters: Optional[Union[Parameter, Parameters, Sequence[Parameter]]] = None,
        analytic_integral: Optional[Callable] = None,
        name: Optional[str] = None,
    ):
        """Initialize an analytic function.

        Args:
            domain: Input space of the function
            codomain: Output space of the function
            func: The function implementation
            parameters: Parameters of the function
            analytic_integral: The analytic integral implementation
            name: Name of the function
        """
        super().__init__(
            domain=domain, codomain=codomain, parameters=parameters, name=name
        )
        self._func = func
        self._analytic_integral = analytic_integral

        # Set properties based on provided information
        if analytic_integral is not None:
            self.properties.has_analytic_integral = True

    def _call_impl(self, x, parameters=None, **kwargs) -> Any:
        """Implementation of the function call."""
        # Prepare parameters
        param_values = {}
        if parameters is not None:
            param_values = parameters
        else:
            for param in self.parameters.params:
                param_values[param.name] = param.value

        # Call the function
        # Handle both dictionary and non-dictionary inputs
        if isinstance(x, dict):
            return self._func(x, **param_values, **kwargs)
        else:
            # Convert to dictionary if domain has a single variable
            if len(self.domain.variables) == 1:
                var_name = self.domain.variables[0].name
                return self._func({var_name: x}, **param_values, **kwargs)
            else:
                # For multiple variables, assume x is a tuple or list of values
                var_dict = {var.name: val for var, val in zip(self.domain.variables, x)}
                return self._func(var_dict, **param_values, **kwargs)

    def integral(self, limits, parameters=None, **kwargs) -> Any:
        """Calculate the integral of the function over the specified limits."""
        if (
            self.properties.has_analytic_integral
            and self._analytic_integral is not None
        ):
            # Prepare parameters
            param_values = {}
            if parameters is not None:
                param_values = parameters
            else:
                for param in self.parameters.params:
                    param_values[param.name] = param.value

            # Calculate analytic integral
            return self._analytic_integral(limits, **param_values, **kwargs)
        else:
            # Fall back to numerical integration
            from zfit2.integration import integrate_numerically

            return integrate_numerically(self, limits, parameters=parameters, **kwargs)


class ComposedFunc(Func):
    """A function composed of two functions: f(g(x))."""

    def __init__(self, outer_func: Func, inner_func: Func):
        """Initialize a composed function.

        Args:
            outer_func: The outer function f
            inner_func: The inner function g
        """
        if not isinstance(inner_func.codomain, Variables) or not isinstance(
            outer_func.domain, Variables
        ):
            raise TypeError(
                "Both inner function codomain and outer function domain must be Variables"
            )

        # Check compatibility
        if len(inner_func.codomain.variables) != len(outer_func.domain.variables):
            raise ValueError(
                "Incompatible function composition: codomain and domain dimensions don't match"
            )

        # Create combined parameter list
        combined_params = Parameters(
            inner_func.parameters.params + outer_func.parameters.params
        )

        super().__init__(
            domain=inner_func.domain,
            codomain=outer_func.codomain,
            parameters=combined_params,
            name=f"{outer_func.name}_{inner_func.name}",
        )

        self.outer_func = outer_func
        self.inner_func = inner_func

        # Set properties based on component functions
        self._update_properties()

    def _update_properties(self):
        """Update properties based on component functions."""
        # A composed function is linear if both components are linear
        self.properties.is_linear = (
            self.outer_func.properties.is_linear
            and self.inner_func.properties.is_linear
        )

        # A composed function has analytic integral under certain conditions
        # For now, we'll be conservative
        self.properties.has_analytic_integral = False

    def _call_impl(self, x, parameters=None, **kwargs) -> Any:
        """Implementation of the function call."""
        # Split parameters for inner and outer functions
        inner_params = {}
        outer_params = {}

        if parameters is not None:
            # Assign parameters to the correct function
            for name, value in parameters.items():
                for param in self.inner_func.parameters.params:
                    if param.name == name:
                        inner_params[name] = value

                for param in self.outer_func.parameters.params:
                    if param.name == name:
                        outer_params[name] = value

        # Apply inner function
        inner_result = self.inner_func(x, parameters=inner_params, **kwargs)

        # Apply outer function
        return self.outer_func(inner_result, parameters=outer_params, **kwargs)


class PolynomialFunc(AnalyticFunc):
    """A polynomial function."""

    def __init__(
        self,
        domain: Union[Variable, Variables, Sequence[Variable]],
        coefficients: Union[Parameter, Parameters, Sequence[Parameter]],
        degree: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """Initialize a polynomial function.

        Args:
            domain: Input space of the function
            coefficients: Coefficients of the polynomial
            degree: Degree of the polynomial (inferred from coefficients if None)
            name: Name of the function
        """
        # Check that domain is 1D
        if isinstance(domain, Variables) and len(domain.variables) != 1:
            raise ValueError("PolynomialFunc requires a 1D domain")

        # Convert coefficients to Parameters
        if isinstance(coefficients, Parameter):
            coeffs = Parameters([coefficients])
            self.degree = 1
        elif isinstance(coefficients, Parameters):
            coeffs = coefficients
            self.degree = len(coeffs.params) - 1
        else:
            coeffs = Parameters(list(coefficients))
            self.degree = len(coeffs.params) - 1

        # Override degree if specified
        if degree is not None:
            self.degree = degree

        # Define the polynomial function
        def poly_func(x, **params):
            # Handle dictionary input
            if isinstance(x, dict):
                var_name = domain.variables[0].name
                x_val = x[var_name]
            else:
                x_val = x[0] if isinstance(x, tuple) else x

            # Use JAX-compatible approach to calculate polynomial
            coeffs = [params[param_name] for param_name in sorted(params.keys())]
            result = coeffs[0]  # Constant term

            # Add higher-order terms
            x_power = x_val
            for coeff in coeffs[1:]:
                result = result + coeff * x_power
                x_power = x_power * x_val

            return result

        # Define the analytic integral
        def poly_integral(limits, **params):
            lower, upper = limits[0]
            result = 0.0
            for i, param_name in enumerate(params):
                # Integral of x^i is x^(i+1)/(i+1)
                term = params[param_name] / (i + 1)
                result += term * (upper ** (i + 1) - lower ** (i + 1))
            return result

        super().__init__(
            domain=domain,
            codomain=Variable("y", lower=None, upper=None),
            func=poly_func,
            parameters=coeffs,
            analytic_integral=poly_integral,
            name=name or "Polynomial",
        )

        # Set properties
        self.properties.has_analytic_integral = True
        self.properties.is_linear = True

        # All parameters are linear
        for param in self.parameters.params:
            self.properties.linear_parameters.add(param.name)


class GaussianFunc(AnalyticFunc):
    """A Gaussian function: A * exp(-0.5 * ((x - mu) / sigma)^2)."""

    def __init__(
        self,
        domain: Union[Variable, Variables, Sequence[Variable]],
        mu: Parameter,
        sigma: Parameter,
        amplitude: Optional[Parameter] = None,
        name: Optional[str] = None,
    ):
        """Initialize a Gaussian function.

        Args:
            domain: Input space of the function
            mu: Mean parameter
            sigma: Standard deviation parameter
            amplitude: Amplitude parameter
            name: Name of the function
        """
        # Check that domain is 1D
        if isinstance(domain, Variables) and len(domain.variables) != 1:
            raise ValueError("GaussianFunc requires a 1D domain")

        params = [mu, sigma]
        if amplitude is not None:
            params.append(amplitude)

        # Define the Gaussian function
        def gaussian_func(x, mu, sigma, amplitude=1.0, **kwargs):
            # Handle dictionary input
            if isinstance(x, dict):
                var_name = domain.variables[0].name
                x_val = x[var_name]
            else:
                x_val = x[0] if isinstance(x, tuple) else x

            # Use JAX-compatible approach
            exponent = -0.5 * ((x_val - mu) / sigma) ** 2
            return amplitude * jnp.exp(exponent)

        # Define the analytic integral for 1D
        def gaussian_integral(limits, mu, sigma, amplitude=1.0, **kwargs):
            lower, upper = limits[0]
            # Standard normal CDF
            from scipy import special

            # Standardize limits
            lower_std = (lower - mu) / sigma
            upper_std = (upper - mu) / sigma

            # Calculate integral using error function
            sqrt_2 = jnp.sqrt(2.0)
            integral = 0.5 * (
                special.erf(upper_std / sqrt_2) - special.erf(lower_std / sqrt_2)
            )

            # Multiply by amplitude and scaling factor
            return amplitude * sigma * jnp.sqrt(2 * jnp.pi) * integral

        super().__init__(
            domain=domain,
            codomain=Variable("y", lower=0.0, upper=None),
            func=gaussian_func,
            parameters=Parameters(params),
            analytic_integral=gaussian_integral,
            name=name or "Gaussian",
        )

        # Set properties
        self.properties.has_analytic_integral = True
        self.properties.is_positive = True

        # Amplitude is a linear parameter
        if amplitude is not None:
            self.properties.linear_parameters.add(amplitude.name)


def create_function_from_callable(
    func: Callable,
    domain: Union[Variable, Variables, Sequence[Variable]],
    parameters: Union[Parameter, Parameters, Sequence[Parameter]],
    codomain: Optional[Union[Variable, Variables, Sequence[Variable]]] = None,
    analytic_integral: Optional[Callable] = None,
    name: Optional[str] = None,
) -> AnalyticFunc:
    """Create a function from a callable.

    Args:
        func: The function implementation
        domain: Input space of the function
        parameters: Parameters of the function
        codomain: Output space of the function (defaults to R)
        analytic_integral: The analytic integral implementation
        name: Name of the function

    Returns:
        An AnalyticFunc object
    """
    if codomain is None:
        codomain = Variable("y", lower=None, upper=None)

    return AnalyticFunc(
        domain=domain,
        codomain=codomain,
        func=func,
        parameters=parameters,
        analytic_integral=analytic_integral,
        name=name or func.__name__,
    )
