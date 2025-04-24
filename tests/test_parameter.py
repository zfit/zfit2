from __future__ import annotations

import numpy as np
import pytest

from zfit2.parameter import Parameter, Parameters


def test_parameter_creation():
    param = Parameter("test", value=1.0, lower=0.0, upper=2.0)
    assert param.name == "test"
    assert param.value == 1.0
    assert param.lower == 0.0
    assert param.upper == 2.0
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
    np.testing.assert_array_equal(params.stepsizes(), np.array([0.1, 0.2]))


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
