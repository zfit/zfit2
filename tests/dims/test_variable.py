from __future__ import annotations

import numpy as np
import pytest

from zfit2.dims.parameter import Parameter, Parameters
from zfit2.dims.variable import Variable, Variables


def test_variable_creation():
    var = Variable("x", lower=-1.0, upper=1.0)
    assert var.name == "x"
    assert var.lower == pytest.approx(-1.0)
    assert var.upper == pytest.approx(1.0)
    assert var.label == "x"


def test_variable_with_label():
    var = Variable("x", label="Position", lower=-1.0, upper=1.0)
    assert var.label == "Position"


def test_variable_to_vars():
    var = Variable("x", lower=-1.0, upper=1.0)
    vars_collection = var.to_vars()
    assert isinstance(vars_collection, Variables)
    assert len(vars_collection.variables) == 1
    assert vars_collection.variables[0] == var


def test_variables_creation():
    var1 = Variable("x", lower=-1.0, upper=1.0)
    var2 = Variable("y", lower=0.0, upper=2.0)
    vars_collection = Variables([var1, var2])
    assert len(vars_collection.variables) == 2
    assert vars_collection.variables[0] == var1
    assert vars_collection.variables[1] == var2


def test_variables_single_input():
    var = Variable("x", lower=-1.0, upper=1.0)
    vars_collection = Variables(var)
    assert len(vars_collection.variables) == 1
    assert vars_collection.variables[0] == var


def test_variables_properties():
    var1 = Variable("x", label="Position", lower=-1.0, upper=1.0)
    var2 = Variable("y", label="Height", lower=0.0, upper=2.0)
    vars_collection = Variables([var1, var2])

    assert vars_collection.names == ["x", "y"]
    assert vars_collection.labels == ["Position", "Height"]
    np.testing.assert_array_equal(vars_collection.lowers, np.array([-1.0, 0.0]))
    np.testing.assert_array_equal(vars_collection.uppers, np.array([1.0, 2.0]))


def test_variables_to_vars():
    var1 = Variable("x", lower=-1.0, upper=1.0)
    var2 = Variable("y", lower=0.0, upper=2.0)
    vars_collection = Variables([var1, var2])
    assert vars_collection.to_vars() is vars_collection


def test_parameter_inheritance():
    param = Parameter("p", value=1.0, lower=0.0, upper=2.0)
    assert isinstance(param, Variable)
    assert param.name == "p"
    assert param.lower == pytest.approx(0.0)
    assert param.upper == pytest.approx(2.0)


def test_parameters_inheritance():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    params = Parameters([param1, param2])
    assert isinstance(params, Variables)
    assert params.names == ["p1", "p2"]
    np.testing.assert_array_equal(params.lowers, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(params.uppers, np.array([2.0, 3.0]))


@pytest.mark.parametrize(
    ("name", "lower", "upper", "label"),
    [
        ("x", -1.0, 1.0, None),
        ("y", 0.0, 10.0, "Y axis"),
        ("z", -np.inf, np.inf, "Z parameter"),
        ("theta", 0.0, 2 * np.pi, "Angle"),
    ],
)
def test_variable_properties(name, lower, upper, label):
    """Test variable creation with various configurations."""
    var = Variable(name, lower=lower, upper=upper, label=label)
    assert var.name == name
    assert var.lower == pytest.approx(lower) if np.isfinite(lower) else lower
    assert var.upper == pytest.approx(upper) if np.isfinite(upper) else upper
    assert var.label == (label if label is not None else name)


@pytest.mark.parametrize("n_vars", [1, 2, 5, 10])
def test_variables_collection_sizes(n_vars):
    """Test Variables collection with different sizes."""
    vars_list = [Variable(f"var_{i}", lower=i, upper=i + 1) for i in range(n_vars)]
    vars_collection = Variables(vars_list)

    assert len(vars_collection.variables) == n_vars
    assert len(vars_collection.names) == n_vars
    assert len(vars_collection.lowers) == n_vars
    assert len(vars_collection.uppers) == n_vars

    # Check values
    expected_lowers = np.arange(n_vars, dtype=float)
    expected_uppers = np.arange(1, n_vars + 1, dtype=float)
    np.testing.assert_array_equal(vars_collection.lowers, expected_lowers)
    np.testing.assert_array_equal(vars_collection.uppers, expected_uppers)
