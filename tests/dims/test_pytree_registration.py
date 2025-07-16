"""Test JAX PyTree registration for Variables and Parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zfit2.dims.parameter import Parameter, Parameters
from zfit2.dims.variable import Variable, Variables


class TestVariablePyTree:
    """Test Variable PyTree registration."""

    def test_variable_pytree(self):
        """Test Variable PyTree registration."""
        var = Variable("x", label="X variable", lower=-10, upper=10)

        # Test tree flattening and unflattening
        flat, treedef = jax.tree_util.tree_flatten(var)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        assert reconstructed.name == var.name
        assert reconstructed.label == var.label
        assert reconstructed.lower == pytest.approx(var.lower)
        assert reconstructed.upper == pytest.approx(var.upper)

    def test_variables_pytree(self):
        """Test Variables PyTree registration."""
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


class TestParameterPyTree:
    """Test Parameter PyTree registration."""

    def test_parameter_pytree(self):
        """Test Parameter PyTree registration."""
        param = Parameter(
            "alpha", value=1.5, lower=0, upper=5, stepsize=0.1, fixed=False
        )

        # Test tree flattening and unflattening
        flat, treedef = jax.tree_util.tree_flatten(param)
        reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

        assert reconstructed.name == param.name
        assert reconstructed.value == pytest.approx(param.value)
        assert reconstructed.lower == pytest.approx(param.lower)
        assert reconstructed.upper == pytest.approx(param.upper)
        assert reconstructed.stepsize == pytest.approx(param.stepsize)
        assert reconstructed.fixed == param.fixed

    def test_parameters_pytree(self):
        """Test Parameters PyTree registration."""
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
        assert reconstructed.params[1].fixed is True


class TestJaxTransformations:
    """Test JAX transformations with Variables and Parameters."""

    def test_gradient_computation_with_parameters(self):
        """Test JAX gradient computation with Parameters."""

        def loss_fn(params):
            # Simulate a loss function that uses parameter values
            return jnp.sum(jnp.array([p.value for p in params.params]) ** 2)

        # Create Parameters
        param1 = Parameter("a", value=1.0)
        param2 = Parameter("b", value=2.0)
        params = Parameters([param1, param2])

        # Test gradient computation
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # The gradients should be with respect to the parameter values
        expected_grads = np.array([2.0, 4.0])  # 2*value for each parameter
        actual_grads = np.array([g.value for g in grads.params])

        np.testing.assert_allclose(actual_grads, expected_grads)

    def test_vmap_with_variables(self):
        """Test JAX vmap with Variables."""

        def process_var(lower, upper):
            # Process the bounds
            return jnp.array([lower, upper])

        # Create variables with specific bounds
        lowers = jnp.array([-1.0, -2.0, -3.0])
        uppers = jnp.array([1.0, 2.0, 3.0])

        # vmap over the bounds
        vmapped_fn = jax.vmap(process_var)
        result = vmapped_fn(lowers, uppers)

        expected = np.array([[-1, 1], [-2, 2], [-3, 3]])
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        ("value1", "value2"),
        [
            (1.0, 2.0),
            (0.5, 1.5),
            (-1.0, 3.0),
        ],
    )
    def test_jit_compilation_with_parameters(self, value1, value2):
        """Test JAX JIT compilation with Parameters."""

        def compute_sum(params):
            return jnp.sum(jnp.array([p.value for p in params.params]))

        param1 = Parameter("a", value=value1)
        param2 = Parameter("b", value=value2)
        params = Parameters([param1, param2])

        # Test JIT compilation
        jitted_fn = jax.jit(compute_sum)
        result = jitted_fn(params)

        expected = value1 + value2
        assert result == pytest.approx(expected)
