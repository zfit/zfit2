"""Tests for the function module."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from zfit2.func import (
    Func, AnalyticFunc, ComposedFunc, PolynomialFunc, GaussianFunc, create_function_from_callable
)
from zfit2.parameter import Parameter, Parameters
from zfit2.variable import Variable, Variables


def test_func_init():
    """Test initialization of a function."""
    # Create a function
    x = Variable("x", lower=0, upper=10)
    y = Variable("y", lower=0, upper=10)
    p1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    p2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    
    func = Func(domain=x, codomain=y, parameters=[p1, p2], name="TestFunc")
    
    # Check attributes
    assert func.name == "TestFunc"
    assert len(func.parameters.params) == 2
    assert func.parameters.params[0] == p1
    assert func.parameters.params[1] == p2
    assert func.domain.variables[0] == x
    assert func.codomain.variables[0] == y


def test_analytic_func():
    """Test AnalyticFunc class."""
    # Create variables and parameters
    x = Variable("x", lower=0, upper=10)
    y = Variable("y", lower=None, upper=None)
    a = Parameter("a", value=2.0, lower=0.0, upper=5.0)
    b = Parameter("b", value=3.0, lower=0.0, upper=5.0)
    
    # Define a simple quadratic function: f(x) = a*x^2 + b*x
    def quad_func(x, a, b):
        return a * x["x"] ** 2 + b * x["x"]
    
    # Define its analytic integral
    def quad_integral(limits, a, b):
        lower, upper = limits[0]
        return (a / 3) * (upper ** 3 - lower ** 3) + (b / 2) * (upper ** 2 - lower ** 2)
    
    # Create the function
    func = AnalyticFunc(
        domain=x,
        codomain=y,
        func=quad_func,
        parameters=[a, b],
        analytic_integral=quad_integral,
        name="Quadratic"
    )
    
    # Test function evaluation
    result = func({"x": 2.0})
    assert np.isclose(result, 2.0 * 2.0 ** 2 + 3.0 * 2.0)
    
    # Test analytic integral
    integral = func.integral([(0.0, 1.0)])
    assert np.isclose(integral, (2.0 / 3) * (1.0 ** 3 - 0.0 ** 3) + (3.0 / 2) * (1.0 ** 2 - 0.0 ** 2))


def test_composed_func():
    """Test function composition."""
    # Create variables and parameters
    x = Variable("x", lower=0, upper=10)
    y = Variable("y", lower=None, upper=None)
    z = Variable("z", lower=None, upper=None)
    
    a = Parameter("a", value=2.0, lower=0.0, upper=5.0)
    b = Parameter("b", value=3.0, lower=0.0, upper=5.0)
    c = Parameter("c", value=1.5, lower=0.0, upper=5.0)
    
    # Define inner function: g(x) = a*x + b
    def inner_func(x, a, b):
        return a * x["x"] + b
    
    # Define outer function: f(y) = c*y^2
    def outer_func(y, c):
        return c * y["y"] ** 2
    
    # Create the functions
    inner = AnalyticFunc(
        domain=x,
        codomain=y,
        func=inner_func,
        parameters=[a, b],
        name="Linear"
    )
    
    outer = AnalyticFunc(
        domain=y,
        codomain=z,
        func=outer_func,
        parameters=[c],
        name="Quadratic"
    )
    
    # Compose the functions: f(g(x)) = c*(a*x + b)^2
    composed = outer.compose(inner)
    
    # Test composition
    result = composed({"x": 2.0})
    expected = 1.5 * (2.0 * 2.0 + 3.0) ** 2
    assert np.isclose(result, expected)


def test_polynomial_func():
    """Test polynomial function."""
    # Create variable and parameters
    x = Variable("x", lower=-5, upper=5)
    p0 = Parameter("p0", value=1.0, lower=-5.0, upper=5.0)
    p1 = Parameter("p1", value=2.0, lower=-5.0, upper=5.0)
    p2 = Parameter("p2", value=3.0, lower=-5.0, upper=5.0)
    
    # Create polynomial function: f(x) = 1 + 2*x + 3*x^2
    poly = PolynomialFunc(
        domain=x,
        coefficients=[p0, p1, p2],
        name="Polynomial"
    )
    
    # Test function evaluation
    result = poly({"x": 2.0})
    expected = 1.0 + 2.0 * 2.0 + 3.0 * 2.0 ** 2
    assert np.isclose(result, expected)
    
    # Test integral
    integral = poly.integral([(0.0, 1.0)])
    expected = 1.0 * 1.0 + 2.0 * 1.0 ** 2 / 2 + 3.0 * 1.0 ** 3 / 3
    assert np.isclose(integral, expected)


def test_gaussian_func():
    """Test Gaussian function."""
    # Create variable and parameters
    x = Variable("x", lower=-10, upper=10)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=5.0)
    amplitude = Parameter("amplitude", value=2.0, lower=0.1, upper=10.0)
    
    # Create Gaussian function
    gaussian = GaussianFunc(
        domain=x,
        mu=mu,
        sigma=sigma,
        amplitude=amplitude,
        name="Gaussian"
    )
    
    # Test function evaluation
    result = gaussian({"x": 0.0})
    expected = 2.0 * np.exp(-0.5 * ((0.0 - 0.0) / 1.0) ** 2)
    assert np.isclose(result, expected)
    
    # Test integral over symmetric range
    integral = gaussian.integral([(-5.0, 5.0)])
    expected = 2.0 * 1.0 * np.sqrt(2 * np.pi) * (0.5 * (1.0 - (-1.0)))
    assert np.isclose(integral, expected, rtol=1e-5)


def test_create_function_from_callable():
    """Test creating a function from a callable."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=10)
    a = Parameter("a", value=2.0, lower=0.0, upper=5.0)
    b = Parameter("b", value=3.0, lower=0.0, upper=5.0)
    
    # Define a function
    def func(x, a, b):
        return a * x["x"] ** 2 + b * x["x"]
    
    # Define its analytic integral
    def integral(limits, a, b):
        lower, upper = limits[0]
        return (a / 3) * (upper ** 3 - lower ** 3) + (b / 2) * (upper ** 2 - lower ** 2)
    
    # Create the function
    func_obj = create_function_from_callable(
        func=func,
        domain=x,
        parameters=[a, b],
        analytic_integral=integral,
        name="CustomFunc"
    )
    
    # Test function evaluation
    result = func_obj({"x": 2.0})
    assert np.isclose(result, 2.0 * 2.0 ** 2 + 3.0 * 2.0)
    
    # Test analytic integral
    integral_result = func_obj.integral([(0.0, 1.0)])
    expected = (2.0 / 3) * (1.0 ** 3 - 0.0 ** 3) + (3.0 / 2) * (1.0 ** 2 - 0.0 ** 2)
    assert np.isclose(integral_result, expected)


def test_jit_compatibility():
    """Test JIT compatibility of functions."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=10)
    a = Parameter("a", value=2.0, lower=0.0, upper=5.0)
    b = Parameter("b", value=3.0, lower=0.0, upper=5.0)
    
    # Create polynomial function
    poly = PolynomialFunc(
        domain=x,
        coefficients=[a, b],
        name="Polynomial"
    )
    
    # Test JIT compilation
    jitted_func = jax.jit(lambda x_val: poly({"x": x_val}))
    
    # Test function evaluation
    result = jitted_func(2.0)
    expected = 2.0 + 3.0 * 2.0
    assert np.isclose(result, expected)