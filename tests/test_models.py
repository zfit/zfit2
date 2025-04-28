"""Tests for the models module."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from zfit2.func import PolynomialFunc, GaussianFunc
from zfit2.parameter import Parameter, Parameters
from zfit2.variable import Variable, Variables
from zfit2.models import Model, GaussianModel, PolynomialModel
from zfit2.integration import integrate


def test_model_creation():
    """Test creation of a model."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=10)
    p0 = Parameter("p0", value=1.0, lower=-5.0, upper=5.0)
    p1 = Parameter("p1", value=2.0, lower=-5.0, upper=5.0)
    
    # Create polynomial function
    poly = PolynomialFunc(
        domain=x,
        coefficients=[p0, p1],
        name="Polynomial"
    )
    
    # Create model
    model = Model(
        func=poly,
        domain=x,
        name="PolyModel"
    )
    
    # Check attributes
    assert model.name == "PolyModel"
    assert model.func == poly
    assert model.domain.variables[0] == x
    assert len(model.params.params) == 2
    assert model.params.params[0] == p0
    assert model.params.params[1] == p1


def test_model_pdf():
    """Test PDF calculation."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=1)
    p0 = Parameter("p0", value=1.0, lower=-5.0, upper=5.0)
    p1 = Parameter("p1", value=2.0, lower=-5.0, upper=5.0)
    
    # Create polynomial function: f(x) = 1 + 2*x
    poly = PolynomialFunc(
        domain=x,
        coefficients=[p0, p1],
        name="Polynomial"
    )
    
    # Create model
    model = Model(
        func=poly,
        domain=x,
        name="PolyModel"
    )
    
    # Calculate normalization factor
    norm_factor = integrate(poly, [(0.0, 1.0)])
    
    # Test PDF at a point
    point = {"x": 0.5}
    expected_pdf = poly(point) / norm_factor
    actual_pdf = model.pdf(point)
    
    assert np.isclose(actual_pdf, expected_pdf, rtol=1e-5)
    
    # Test PDF with custom parameters
    params = {"p0": 3.0, "p1": 4.0}
    norm_factor_custom = integrate(poly, [(0.0, 1.0)], parameters=params)
    expected_pdf_custom = poly(point, parameters=params) / norm_factor_custom
    actual_pdf_custom = model.pdf(point, params=params)
    
    assert np.isclose(actual_pdf_custom, expected_pdf_custom, rtol=1e-5)


def test_model_integration():
    """Test model integration."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=1)
    p0 = Parameter("p0", value=1.0, lower=-5.0, upper=5.0)
    p1 = Parameter("p1", value=2.0, lower=-5.0, upper=5.0)
    
    # Create polynomial function: f(x) = 1 + 2*x
    poly = PolynomialFunc(
        domain=x,
        coefficients=[p0, p1],
        name="Polynomial"
    )
    
    # Create model
    model = Model(
        func=poly,
        domain=x,
        name="PolyModel"
    )
    
    # Test integration over full range
    result1 = model.integrate([(0.0, 1.0)])
    assert np.isclose(result1, 1.0, rtol=1e-5)  # Normalized, should be 1.0
    
    # Test integration over partial range
    result2 = model.integrate([(0.0, 0.5)])
    
    # Calculate expected result
    func_integral = integrate(poly, [(0.0, 0.5)])
    norm_factor = integrate(poly, [(0.0, 1.0)])
    expected2 = func_integral / norm_factor
    
    assert np.isclose(result2, expected2, rtol=1e-5)
    
    # Test with custom parameters
    params = {"p0": 3.0, "p1": 4.0}
    result3 = model.integrate([(0.25, 0.75)], params=params)
    
    # Calculate expected result
    func_integral = integrate(poly, [(0.25, 0.75)], parameters=params)
    norm_factor = integrate(poly, [(0.0, 1.0)], parameters=params)
    expected3 = func_integral / norm_factor
    
    assert np.isclose(result3, expected3, rtol=1e-5)


def test_gaussian_model():
    """Test GaussianModel class."""
    # Create variable and parameters
    x = Variable("x", lower=-10, upper=10)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=5.0)
    
    # Create Gaussian model
    model = GaussianModel(
        domain=x,
        mu=mu,
        sigma=sigma,
        name="Gaussian"
    )
    
    # Test PDF at mean
    point = {"x": 0.0}
    pdf_at_mean = model.pdf(point)
    
    # PDF at mean should be 1/sqrt(2*pi*sigma^2) ≈ 0.3989/sigma
    expected_pdf = 0.3989 / 1.0
    
    assert np.isclose(pdf_at_mean, expected_pdf, rtol=1e-2)
    
    # Test integration over a symmetric range
    result = model.integrate([(-5.0, 5.0)])
    # Should be very close to 1 (about 0.9973)
    assert np.isclose(result, 0.9973, rtol=1e-2)


def test_polynomial_model():
    """Test PolynomialModel class."""
    # Create variable and parameters
    x = Variable("x", lower=0, upper=1)
    p0 = Parameter("p0", value=1.0, lower=-5.0, upper=5.0)
    p1 = Parameter("p1", value=2.0, lower=-5.0, upper=5.0)
    
    # Create polynomial model
    model = PolynomialModel(
        domain=x,
        coefficients=[p0, p1],
        name="Polynomial"
    )
    
    # Test PDF at a point
    point = {"x": 0.5}
    pdf = model.pdf(point)
    
    # Calculate expected PDF
    func_value = 1.0 + 2.0 * 0.5
    norm_factor = 1.0 + 2.0 / 2.0  # Integral of 1 + 2x from 0 to 1
    expected_pdf = func_value / norm_factor
    
    assert np.isclose(pdf, expected_pdf, rtol=1e-5)
    
    # Test integration
    result = model.integrate([(0.0, 1.0)])
    assert np.isclose(result, 1.0, rtol=1e-5)  # Normalized, should be 1.0
