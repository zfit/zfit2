from __future__ import annotations

import numpy as np
import pytest

from zfit2.backend import numpy as znp
from zfit2.dist import Distribution
from zfit2.likelihood import NLL, NLLS, OptimizedNLL, SumMethod, ZeroHandler
from zfit2.parameter import Parameter, Parameters
from zfit2.valueholder import ValueHolder
from zfit2.variable import Variable, Variables


class MockDistribution(Distribution):
    """A mock distribution for testing the log-likelihood module."""

    def __init__(self, domain, params, *, name=None, label=None):
        """Initialize the mock distribution."""
        super().__init__(domain=domain, params=params, name=name, label=label)

    def log_pdf(self, x, *, params=None, norm=None):
        """
        Calculate log PDF for a simple Gaussian distribution.

        This is a single-variable Gaussian with parameters mu and sigma.
        """
        if params is None:
            mu = self.params.params[0].value
            sigma = self.params.params[1].value
        else:
            mu = params.get('mu', self.params.params[0].value)
            sigma = params.get('sigma', self.params.params[1].value)

        var_name = self.domain.variables[0].name
        x_val = x[var_name]

        # Gaussian log PDF: -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2π)
        return -0.5 * ((x_val - mu) / sigma) ** 2 - znp.log(sigma) - 0.5 * znp.log(2 * znp.pi)


def test_nll_creation():
    """Test creating a NLL object."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.5, 1.0, -0.5, 0.0, 2.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Check attributes
    assert nll.distribution == dist
    assert isinstance(nll.data, ValueHolder)
    assert nll.name == "NLL_mock_dist"


def test_pointwise_loglik():
    """Test calculating pointwise log-likelihood."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Calculate pointwise log-likelihood
    pointwise = nll.pointwise_loglik()

    # For x = 0, mu = 0, sigma = 1, the log PDF should be -0.5 * log(2π) ≈ -0.9189
    # For x = 1, mu = 0, sigma = 1, the log PDF should be -0.5 - 0.5 * log(2π) ≈ -1.4189
    # For x = -1, mu = 0, sigma = 1, the log PDF should be -0.5 - 0.5 * log(2π) ≈ -1.4189
    expected = np.array([-0.5 * np.log(2 * np.pi), 
                         -0.5 - 0.5 * np.log(2 * np.pi),
                         -0.5 - 0.5 * np.log(2 * np.pi)])

    # Check results
    np.testing.assert_allclose(pointwise, expected, rtol=1e-5)


def test_sum_loglik():
    """Test calculating the sum of log-likelihoods with different methods."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Calculate sum of log-likelihoods with different methods
    sum_direct = nll.sum_loglik(method=SumMethod.DIRECT)
    sum_stable = nll.sum_loglik(method=SumMethod.STABLE)
    sum_pairwise = nll.sum_loglik(method=SumMethod.PAIRWISE)
    sum_kahan = nll.sum_loglik(method=SumMethod.KAHAN)
    sum_neumaier = nll.sum_loglik(method=SumMethod.NEUMAIER)

    # The expected sum is approximately -3.7568
    expected = -0.5 * np.log(2 * np.pi) + 2 * (-0.5 - 0.5 * np.log(2 * np.pi))

    # Check results
    np.testing.assert_allclose(sum_direct, expected, rtol=1e-5)
    np.testing.assert_allclose(sum_stable, expected, rtol=1e-5)
    np.testing.assert_allclose(sum_pairwise, expected, rtol=1e-5)
    np.testing.assert_allclose(sum_kahan, expected, rtol=1e-5)
    np.testing.assert_allclose(sum_neumaier, expected, rtol=1e-5)

    # Try to use logsumexp - this will change the semantic meaning, so don't compare to expected
    try:
        sum_logsumexp = nll.sum_loglik(method=SumMethod.LOGSUMEXP)
        assert sum_logsumexp is not None
    except (ImportError, AttributeError):
        pass  # If JAX or scipy is not available, skip this test


def test_nll_calculate():
    """Test calculating the negative log-likelihood."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Calculate NLL
    result = nll.nll()

    # The expected NLL is the negative of the expected sum, approximately 3.7568
    expected = -(-0.5 * np.log(2 * np.pi) + 2 * (-0.5 - 0.5 * np.log(2 * np.pi)))

    # Check result
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_zero_handling_methods():
    """Test different zero handling methods."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data with extreme values that will produce negative log-likelihoods
    data = {"x": np.array([10.0, 20.0, -15.0])}  # These will produce very negative log-likelihoods

    # Test different zero handling methods
    epsilon_nll = NLL(distribution=dist, data=data, zero_handling=ZeroHandler.EPSILON, epsilon=1e-5)
    clamp_nll = NLL(distribution=dist, data=data, zero_handling=ZeroHandler.CLAMP, min_value=-100.0)
    smoothed_nll = NLL(distribution=dist, data=data, zero_handling=ZeroHandler.SMOOTHED, min_value=-100.0)
    shifted_nll = NLL(distribution=dist, data=data, zero_handling=ZeroHandler.SHIFTED, epsilon=1e-5)

    # Calculate NLL with different handling methods
    epsilon_result = epsilon_nll.nll()
    clamp_result = clamp_nll.nll()
    smoothed_result = smoothed_nll.nll()
    shifted_result = shifted_nll.nll()

    # All should produce finite results
    assert np.isfinite(epsilon_result)
    assert np.isfinite(clamp_result)
    assert np.isfinite(smoothed_result)
    assert np.isfinite(shifted_result)

    # The clamp method should produce a value ≥ -300 (3 values, each clamped at -100)
    assert clamp_result >= -300.0

    # The smoothed method should also be bounded, but may have values below clamp bound
    assert smoothed_result > -znp.inf


def test_nll_with_params():
    """Test NLL calculation with provided parameters."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Calculate NLL with different parameters
    result1 = nll.nll({"mu": 0.5, "sigma": 1.5})

    # Check that the result is different
    assert result1 != nll.nll()


def test_nll_gradient():
    """Test calculating the gradient of the NLL."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object with numerical gradient
    nll = NLL(distribution=dist, data=data, use_jax=False)

    # Calculate gradient
    gradient = nll.grad()

    # Check that gradient is a ValueHolder with the right keys
    assert isinstance(gradient, ValueHolder)
    assert "mu" in gradient
    assert "sigma" in gradient


def test_nll_hessian():
    """Test calculating the Hessian of the NLL."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object with numerical Hessian
    nll = NLL(distribution=dist, data=data, use_jax=False)

    # Calculate Hessian
    hessian = nll.hess()

    # Check that Hessian is a ValueHolder with the right keys
    assert isinstance(hessian, ValueHolder)
    assert "mu" in hessian
    assert "sigma" in hessian
    assert "mu" in hessian["mu"]
    assert "sigma" in hessian["mu"]
    assert "mu" in hessian["sigma"]
    assert "sigma" in hessian["sigma"]


def test_nll_hessian_vector_product():
    """Test calculating the Hessian-vector product of the NLL."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object with numerical methods
    nll = NLL(distribution=dist, data=data, use_jax=False)

    # Create a vector
    vector = {"mu": 1.0, "sigma": 0.5}

    # Calculate Hessian-vector product
    hvp = nll.hvp(vector)

    # Check that HVP is a ValueHolder with the right keys
    assert isinstance(hvp, ValueHolder)
    assert "mu" in hvp
    assert "sigma" in hvp


def test_value_and_gradient():
    """Test calculating both value and gradient."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data, use_jax=False)

    # Calculate value and gradient
    value, gradient = nll.val_and_grad()

    # Check results
    # JAX arrays are returned instead of Python floats
    assert isinstance(gradient, ValueHolder)
    assert "mu" in gradient
    assert "sigma" in gradient

    # The value should match the result of nll()
    np.testing.assert_allclose(value, nll.nll(), rtol=1e-10)


def test_nlls_creation():
    """Test creating a NLLS object."""
    # Create variables and parameters
    x1 = Variable("x1", lower=-10.0, upper=10.0)
    mu1 = Parameter("mu1", value=0.0, lower=-5.0, upper=5.0)
    sigma1 = Parameter("sigma1", value=1.0, lower=0.1, upper=10.0)

    x2 = Variable("x2", lower=-10.0, upper=10.0)
    mu2 = Parameter("mu2", value=0.0, lower=-5.0, upper=5.0)
    sigma2 = Parameter("sigma2", value=1.0, lower=0.1, upper=10.0)

    # Create domains and parameters collections
    domain1 = Variables(x1)
    params1 = Parameters([mu1, sigma1])

    domain2 = Variables(x2)
    params2 = Parameters([mu2, sigma2])

    # Create mock distributions
    dist1 = MockDistribution(domain=domain1, params=params1, name="mock_dist1")
    dist2 = MockDistribution(domain=domain2, params=params2, name="mock_dist2")

    # Create data
    data1 = {"x1": np.array([0.0, 1.0, -1.0])}
    data2 = {"x2": np.array([0.5, -0.5, 1.5])}

    # Create NLL objects
    nll1 = NLL(distribution=dist1, data=data1)
    nll2 = NLL(distribution=dist2, data=data2)

    # Create a NLLS object
    nlls = NLLS([nll1, nll2])

    # Check attributes
    assert len(nlls.likelihoods) == 2
    assert nlls.likelihoods[0] == nll1
    assert nlls.likelihoods[1] == nll2


def test_nlls_calculate():
    """Test calculating the combined negative log-likelihood."""
    # Create variables and parameters
    x1 = Variable("x1", lower=-10.0, upper=10.0)
    mu1 = Parameter("mu1", value=0.0, lower=-5.0, upper=5.0)
    sigma1 = Parameter("sigma1", value=1.0, lower=0.1, upper=10.0)

    x2 = Variable("x2", lower=-10.0, upper=10.0)
    mu2 = Parameter("mu2", value=0.0, lower=-5.0, upper=5.0)
    sigma2 = Parameter("sigma2", value=1.0, lower=0.1, upper=10.0)

    # Create domains and parameters collections
    domain1 = Variables(x1)
    params1 = Parameters([mu1, sigma1])

    domain2 = Variables(x2)
    params2 = Parameters([mu2, sigma2])

    # Create mock distributions
    dist1 = MockDistribution(domain=domain1, params=params1, name="mock_dist1")
    dist2 = MockDistribution(domain=domain2, params=params2, name="mock_dist2")

    # Create data
    data1 = {"x1": np.array([0.0, 1.0, -1.0])}
    data2 = {"x2": np.array([0.5, -0.5, 1.5])}

    # Create NLL objects
    nll1 = NLL(distribution=dist1, data=data1)
    nll2 = NLL(distribution=dist2, data=data2)

    # Create a NLLS object
    nlls = NLLS([nll1, nll2])

    # Calculate individual NLLs
    result1 = nll1.nll()
    result2 = nll2.nll()

    # Calculate combined NLL
    result_combined = nlls.nll()

    # Check that the combined result is the sum of individual results
    np.testing.assert_allclose(result_combined, result1 + result2, rtol=1e-10)


def test_optimized_nll():
    """Test the OptimizedNLL class."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create an OptimizedNLL object with global constant subtraction
    opt_nll = OptimizedNLL(
        distribution=dist, 
        data=data, 
        constant_subtraction="auto", 
        epsilon=1e-5,
        sum_method=SumMethod.NEUMAIER
    )

    # Calculate NLL
    result = opt_nll.nll()

    # Create a regular NLL object for comparison
    nll = NLL(distribution=dist, data=data)

    # The optimized NLL should be different from the regular NLL
    # because constant subtraction is applied during initialization
    assert abs(result - nll.nll()) > 1e-10  # First call should already be different
    assert abs(opt_nll.nll() - nll.nll()) > 1e-10  # Second call should also differ

    # Check that the constant is used
    assert opt_nll.constant is not None
    assert opt_nll.epsilon == 1e-5
    assert opt_nll.sum_method == SumMethod.NEUMAIER


def test_optimized_nll_per_point():
    """Test the OptimizedNLL class with per-point constant subtraction."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data with some extreme values
    data = {"x": np.array([0.0, 10.0, -10.0])}

    # Create an OptimizedNLL object with per-point constant subtraction
    opt_nll = OptimizedNLL(
        distribution=dist, 
        data=data, 
        constant_subtraction="per_point",
        zero_handling=ZeroHandler.SMOOTHED,
        epsilon=1e-5
    )

    # Calculate NLL
    result = opt_nll.nll()

    # The first call should compute and store point constants
    assert opt_nll.point_constants is not None

    # Calculate gradient with different summation methods
    grad1 = opt_nll.grad(method=SumMethod.DIRECT)
    grad2 = opt_nll.grad(method=SumMethod.NEUMAIER)

    # The gradients should be the same regardless of summation method
    for key in grad1._values:
        np.testing.assert_allclose(grad1[key], grad2[key], rtol=1e-5)
