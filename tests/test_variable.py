from __future__ import annotations

import numpy as np

from zfit2.parameter import Parameter, Parameters
from zfit2.variable import Variable, Variables


def test_variable_creation():
    var = Variable("x", lower=-1.0, upper=1.0)
    assert var.name == "x"
    assert var.lower == -1.0
    assert var.upper == 1.0
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
    assert param.lower == 0.0
    assert param.upper == 2.0


def test_parameters_inheritance():
    param1 = Parameter("p1", value=1.0, lower=0.0, upper=2.0)
    param2 = Parameter("p2", value=2.0, lower=1.0, upper=3.0)
    params = Parameters([param1, param2])
    assert isinstance(params, Variables)
    assert params.names == ["p1", "p2"]
    np.testing.assert_array_equal(params.lowers, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(params.uppers, np.array([2.0, 3.0]))
