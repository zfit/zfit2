from __future__ import annotations

import jax
import numpy as np
import pytest

from zfit2.dims.parameter import Parameter, Parameters
from zfit2.dims.variable import Variable, Variables


def test_parameter_creation():
    param = Parameter("test", value=1.0, lower=0.0, upper=2.0)
    assert param.name == "test"
    assert param.value == pytest.approx(1.0)
    assert param.lower == pytest.approx(0.0)
    assert param.upper == pytest.approx(2.0)
    assert param.fixed is False
    assert param.stepsize is None
    assert param.prior is None
    assert param.label == "test"


def test_parameter_with_label():
    param = Parameter("test", value=1.0, lower=0.0, upper=2.0, label="Test Param")
    assert param.label == "Test Param"


def test_parameter_to_params():
    param = Parameter("test", value=1.0, lower=0.0, upper=2.0)
    params = param.to_params()
    assert isinstance(params, Parameters)
    assert len(params) == 1
    assert params.params[0] == param


def test_parameters_creation():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    params = Parameters([param1, param2])
    assert len(params) == 2
    assert params.params[0] == param1
    assert params.params[1] == param2


def test_parameters_invalid_creation():
    with pytest.raises(TypeError, match="All elements must be of type Parameter"):
        Parameters([1, 2, 3])


def test_parameters_values():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    params = Parameters([param1, param2])
    np.testing.assert_array_equal(params.values(), np.array([1.0, 2.0]))


def test_parameters_fixed():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0, fixed=True)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0, fixed=False)
    params = Parameters([param1, param2])
    np.testing.assert_array_equal(params.fixed(), np.array([True, False]))


def test_parameters_stepsizes():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0, stepsize=0.1)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0, stepsize=0.2)
    params = Parameters([param1, param2])
    np.testing.assert_array_equal(np.array(params.stepsizes()), np.array([0.1, 0.2]))


def test_parameters_addition():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    param3 = Parameter("p3", value=3.0, lower=2.0, upper=4.0)

    params1 = Parameters([param1])
    params2 = Parameters([param2])

    # Test adding Parameter to Parameters
    result1 = params1 + param3
    assert len(result1) == 2
    assert result1.params == [param1, param3]

    # Test adding Parameters to Parameters
    result2 = params1 + params2
    assert len(result2) == 2
    assert result2.params == [param1, param2]


def test_parameters_invalid_addition():
    params = Parameters([Parameter("p1", value=1.0, lower=0.0, upper=2.0)])
    with pytest.raises(TypeError, match="Can only add Parameter or Parameters objects"):
        params + 1


@pytest.mark.parametrize(
    ("name", "value", "lower", "upper", "stepsize", "fixed"),
    [
        ("alpha", 1.0, 0.0, 2.0, 0.1, False),
        ("beta", 2.5, 1.0, 5.0, 0.2, True),
        ("gamma", 0.0, -1.0, 1.0, 0.05, False),
    ],
)
def test_parameter_properties(name, value, lower, upper, stepsize, fixed):
    """Test parameter creation with various configurations."""
    param = Parameter(
        name, value=value, lower=lower, upper=upper, stepsize=stepsize, fixed=fixed
    )
    assert param.name == name
    assert param.value == pytest.approx(value)
    assert param.lower == pytest.approx(lower)
    assert param.upper == pytest.approx(upper)
    assert param.stepsize == pytest.approx(stepsize)
    assert param.fixed == fixed


def test_variable_pytree():
    """Test Variable JAX PyTree registration."""
    var = Variable("x", label="X variable", lower=-10, upper=10)

    # Test tree flattening and unflattening
    flat, treedef = jax.tree_util.tree_flatten(var)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    assert reconstructed.name == var.name
    assert reconstructed.label == var.label
    assert reconstructed.lower == var.lower
    assert reconstructed.upper == var.upper


def test_variables_pytree():
    """Test Variables JAX PyTree registration."""
    var1 = Variable("x", lower=-10, upper=10)
    var2 = Variable("y", label="Y var", lower=-5, upper=5)
    vars_obj = Variables([var1, var2])

    # Test tree flattening and unflattening
    flat, treedef = jax.tree_util.tree_flatten(vars_obj)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    assert len(reconstructed.variables) == 2
    assert reconstructed.variables[0].name == "x"
    assert reconstructed.variables[1].name == "y"
    assert reconstructed.variables[1].label == "Y var"


def test_parameter_pytree():
    """Test Parameter JAX PyTree registration."""
    param = Parameter("alpha", value=1.5, lower=0, upper=5, stepsize=0.1, fixed=False)

    # Test tree flattening and unflattening
    flat, treedef = jax.tree_util.tree_flatten(param)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    assert reconstructed.name == param.name
    assert reconstructed.value == pytest.approx(param.value)
    assert reconstructed.lower == pytest.approx(param.lower)
    assert reconstructed.upper == pytest.approx(param.upper)
    assert reconstructed.stepsize == pytest.approx(param.stepsize)
    assert reconstructed.fixed == param.fixed


def test_parameters_pytree():
    """Test Parameters JAX PyTree registration."""
    param1 = Parameter("alpha", value=1.5, lower=0, upper=5)
    param2 = Parameter("beta", value=2.0, stepsize=0.2, fixed=True)
    params_obj = Parameters([param1, param2])

    # Test tree flattening and unflattening
    flat, treedef = jax.tree_util.tree_flatten(params_obj)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    assert len(reconstructed.params) == 2
    assert reconstructed.params[0].name == "alpha"
    assert reconstructed.params[0].value == pytest.approx(1.5)
    assert reconstructed.params[1].name == "beta"
    assert reconstructed.params[1].fixed


def test_parameter_jax_gradient():
    """Test JAX gradient computation with Parameters."""

    def loss_fn(params):
        # Simulate a loss function that uses parameter values
        return jax.numpy.sum(jax.numpy.array([p.value for p in params.params]) ** 2)

    # Create Parameters
    param1 = Parameter("a", value=1.0)
    param2 = Parameter("b", value=2.0)
    params = Parameters([param1, param2])

    # Test gradient computation
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)

    # The gradients should be with respect to the parameter values
    expected_grads = [2.0, 4.0]  # 2*value for each parameter
    actual_grads = [g.value for g in grads.params]

    np.testing.assert_allclose(actual_grads, expected_grads)
