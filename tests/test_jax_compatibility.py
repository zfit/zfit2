import pytest
import jax
import jax.numpy as jnp
import numpy as np

from zfit2.data.axis import RegularAxis, VariableAxis, IntegerAxis, CategoryAxis, StrCategoryAxis, IntCategoryAxis, BooleanAxis
from zfit2.parameter import Parameter
from zfit2.variable import Variable
from zfit2.likelihood import NLL, OptimizedNLL, SumMethod
from zfit2.dist import Distribution
from zfit2.data.jax_hist import JaxHist


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
        return -0.5 * ((x_val - mu) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi)


def test_axis_jit_compatibility():
    """Test that axis classes can be used with JAX jit."""
    # Create different types of axes
    regular_axis = RegularAxis(bins=10, start=0, stop=10, name="regular")
    variable_axis = VariableAxis([0, 1, 3, 6, 10], name="variable")
    integer_axis = IntegerAxis(start=0, stop=10, name="integer")
    str_category_axis = StrCategoryAxis(["a", "b", "c"], name="str_category")
    int_category_axis = IntCategoryAxis([1, 2, 3], name="int_category")
    boolean_axis = BooleanAxis(name="boolean")

    # Define a function that uses the axis
    def use_regular_axis(axis, value):
        return axis.index(value)

    def use_variable_axis(axis, value):
        return axis.index(value)

    def use_integer_axis(axis, value):
        return axis.index(value)

    # JIT compile the functions
    jitted_regular = jax.jit(use_regular_axis)
    jitted_variable = jax.jit(use_variable_axis)
    jitted_integer = jax.jit(use_integer_axis)

    # Test the jitted functions
    assert jitted_regular(regular_axis, 5.0) == regular_axis.index(5.0)
    assert jitted_variable(variable_axis, 5.0) == variable_axis.index(5.0)
    assert jitted_integer(integer_axis, 5) == integer_axis.index(5)


def test_parameter_jit_compatibility():
    """Test that Parameter class can be used with JAX jit."""
    # Create a parameter
    param = Parameter("param", value=1.0, lower=0.0, upper=10.0)

    # Define a function that uses the parameter
    def use_parameter(param, factor):
        return param.value * factor

    # JIT compile the function
    jitted_func = jax.jit(use_parameter)

    # Test the jitted function
    assert jitted_func(param, 2.0) == param.value * 2.0


def test_variable_jit_compatibility():
    """Test that Variable class can be used with JAX jit."""
    # Create a variable
    var = Variable("var", lower=0.0, upper=10.0)

    # Define a function that uses the variable
    def use_variable(var, value):
        return jnp.where((value >= var.lower) & (value <= var.upper), value, 0.0)

    # JIT compile the function
    jitted_func = jax.jit(use_variable)

    # Test the jitted function
    assert jitted_func(var, 5.0) == 5.0
    assert jitted_func(var, 15.0) == 0.0


def test_nll_jit_compatibility():
    """Test that NLL class can be used with JAX jit."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    from zfit2.variable import Variables
    from zfit2.parameter import Parameters
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Define a function that uses the NLL
    def use_nll(nll, params):
        return nll.nll(params)

    # JIT compile the function
    jitted_func = jax.jit(use_nll)

    # Test the jitted function
    test_params = {"mu": 0.5, "sigma": 1.5}
    assert jnp.isclose(jitted_func(nll, test_params), nll.nll(test_params))


def test_optimized_nll_jit_compatibility():
    """Test that OptimizedNLL class can be used with JAX jit."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    from zfit2.variable import Variables
    from zfit2.parameter import Parameters
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create an OptimizedNLL object
    opt_nll = OptimizedNLL(
        distribution=dist, 
        data=data, 
        constant_subtraction="auto", 
        epsilon=1e-5,
        sum_method=SumMethod.NEUMAIER
    )

    # Define a function that uses the OptimizedNLL
    def use_opt_nll(opt_nll, params):
        return opt_nll.nll(params)

    # JIT compile the function
    jitted_func = jax.jit(use_opt_nll)

    # Test the jitted function
    test_params = {"mu": 0.5, "sigma": 1.5}
    assert jnp.isclose(jitted_func(opt_nll, test_params), opt_nll.nll(test_params))


def test_jaxhist_jit_compatibility():
    """Test that JaxHist class can be used with JAX jit."""
    # Create axes
    x_axis = RegularAxis(bins=10, start=0, stop=10, name="x")
    y_axis = RegularAxis(bins=10, start=0, stop=10, name="y")

    # Create a JaxHist
    hist = JaxHist(x_axis, y_axis)

    # Fill the histogram
    x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hist.fill([x_data, y_data])

    # Define a function that uses the JaxHist
    def use_hist(hist, x, y):
        indices = [axis.index(val) for axis, val in zip([hist.axes[0], hist.axes[1]], [x, y])]
        return hist._counts[indices[0], indices[1]]

    # JIT compile the function
    jitted_func = jax.jit(use_hist)

    # Test the jitted function
    assert jnp.isclose(jitted_func(hist, 2.5, 2.5), hist._counts[2, 2])


def test_all_optimization_methods():
    """Test all optimization methods with NLL."""
    # Create a variable and parameters
    x = Variable("x", lower=-10.0, upper=10.0)
    mu = Parameter("mu", value=0.0, lower=-5.0, upper=5.0)
    sigma = Parameter("sigma", value=1.0, lower=0.1, upper=10.0)

    # Create a domain and parameters collection
    from zfit2.variable import Variables
    from zfit2.parameter import Parameters
    domain = Variables(x)
    params = Parameters([mu, sigma])

    # Create a mock distribution
    dist = MockDistribution(domain=domain, params=params, name="mock_dist")

    # Create data
    data = {"x": np.array([0.0, 1.0, -1.0])}

    # Create a NLL object
    nll = NLL(distribution=dist, data=data)

    # Test all optimization methods
    methods = [SumMethod.DIRECT, SumMethod.STABLE, SumMethod.PAIRWISE, 
               SumMethod.KAHAN, SumMethod.NEUMAIER, SumMethod.LOGSUMEXP]
    
    for method in methods:
        # Calculate NLL with the method
        result = nll.nll(method=method)
        
        # Define a function that uses the NLL with this method
        def use_nll(nll, params, method=method):
            return nll.nll(params, method=method)
        
        # JIT compile the function
        jitted_func = jax.jit(use_nll)
        
        # Test the jitted function
        test_params = {"mu": 0.5, "sigma": 1.5}
        assert jnp.isclose(jitted_func(nll, test_params), nll.nll(test_params, method=method))