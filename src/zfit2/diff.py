"""Differentiation module for zfit2.

This module provides differentiation functionality for computing
gradients and Hessians of scalar functions, with support for
both JAX-based automatic differentiation and numerical methods.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .backend import numpy as znp
from .valueholder import ValueHolder


def _is_jax_available() -> bool:
    """Check if JAX is available and working."""
    try:
        import jax
        import jax.numpy as jnp

        # Try to perform a simple operation to ensure JAX is working
        jnp.array([1, 2, 3])
        return True
    except (ImportError, ValueError, AttributeError):
        return False


class Differentiator:
    """Base class for differentiation methods."""

    def __init__(self, use_jax: bool = True):
        """Initialize the differentiator.

        Args:
            use_jax: Whether to use JAX for differentiation if available.
        """
        self.use_jax = use_jax and _is_jax_available()

    def grad(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the gradient of a function.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the gradient.

        Returns:
            A ValueHolder containing the gradient values.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def hess(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the Hessian of a function.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the Hessian.

        Returns:
            A ValueHolder containing the Hessian values.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def hvp(
        self, func: Callable, params: dict[str, Any], vector: dict[str, Any]
    ) -> ValueHolder:
        """Compute the Hessian-vector product of a function.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the HVP.
            vector: The vector to multiply with the Hessian.

        Returns:
            A ValueHolder containing the HVP values.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def value_and_grad(
        self, func: Callable, params: dict[str, Any]
    ) -> tuple[float, ValueHolder]:
        """Compute both the function value and its gradient.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate.

        Returns:
            A tuple of (function value, gradient).
        """
        value = func(params)
        gradient = self.grad(func, params)
        return value, gradient


class JaxDifferentiator(Differentiator):
    """JAX-based differentiator for automatic differentiation."""

    def __init__(self):
        """Initialize the JAX differentiator."""
        super().__init__(use_jax=True)
        if not self.use_jax:
            raise ImportError(
                "JAX is not available. Please install JAX or use NumericalDifferentiator."
            )

        import jax

        self.jax = jax

    def _prepare_params(
        self, params: dict[str, Any]
    ) -> tuple[list, list, dict[str, int]]:
        """Prepare parameters for JAX differentiation.

        Args:
            params: The parameter values.

        Returns:
            A tuple of (param_names, param_values, param_indices).
        """
        param_names = list(params.keys())
        param_values = [params[name] for name in param_names]
        param_indices = {name: i for i, name in enumerate(param_names)}
        return param_names, param_values, param_indices

    def _wrap_function(self, func: Callable, param_names: list) -> Callable:
        """Wrap a function to accept a flat list of parameters.

        Args:
            func: The function to wrap.
            param_names: The parameter names.

        Returns:
            A function that accepts a flat list of parameters.
        """

        def wrapped(values):
            params_dict = {
                name: value for name, value in zip(param_names, values, strict=False)
            }
            return func(params_dict)

        return wrapped

    def grad(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the gradient using JAX.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the gradient.

        Returns:
            A ValueHolder containing the gradient values.
        """
        param_names, param_values, _ = self._prepare_params(params)
        wrapped_func = self._wrap_function(func, param_names)

        grad_values = self.jax.grad(wrapped_func)(param_values)

        return ValueHolder(
            {name: value for name, value in zip(param_names, grad_values, strict=False)}
        )

    def hess(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the Hessian using JAX.

        The implementation uses a combination of forward-mode and reverse-mode
        automatic differentiation for efficiency.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the Hessian.

        Returns:
            A ValueHolder containing the Hessian values.
        """
        param_names, param_values, _ = self._prepare_params(params)
        wrapped_func = self._wrap_function(func, param_names)

        # Use jacfwd(jacrev) for efficiency when computing Hessian of scalar function
        hess_func = self.jax.jit(self.jax.jacfwd(self.jax.jacrev(wrapped_func)))
        hess_values = hess_func(param_values)

        # Create nested dictionary for the Hessian
        hess_dict = {}
        for i, name_i in enumerate(param_names):
            hess_dict[name_i] = {}
            for j, name_j in enumerate(param_names):
                hess_dict[name_i][name_j] = hess_values[i][j]

        return ValueHolder(hess_dict)

    def hvp(
        self, func: Callable, params: dict[str, Any], vector: dict[str, Any]
    ) -> ValueHolder:
        """Compute the Hessian-vector product using JAX.

        This is more efficient than computing the full Hessian,
        especially for high-dimensional parameter spaces.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the HVP.
            vector: The vector to multiply with the Hessian.

        Returns:
            A ValueHolder containing the HVP values.
        """
        param_names, param_values, _ = self._prepare_params(params)
        vector_values = [vector.get(name, 0) for name in param_names]
        wrapped_func = self._wrap_function(func, param_names)

        # Forward-over-reverse for efficiency
        def hvp_func(primals, tangents):
            return self.jax.jvp(self.jax.grad(wrapped_func), [primals], [tangents])[1]

        hvp_values = hvp_func(param_values, vector_values)

        return ValueHolder(
            {name: value for name, value in zip(param_names, hvp_values, strict=False)}
        )


class NumericalDifferentiator(Differentiator):
    """Numerical differentiator using finite differences."""

    def __init__(self, step_size: float = 1e-6):
        """Initialize the numerical differentiator.

        Args:
            step_size: Step size for numerical differentiation.
        """
        super().__init__(use_jax=False)
        self.step_size = step_size

    def grad(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the gradient using numerical differentiation.

        Uses central differences for better accuracy.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the gradient.

        Returns:
            A ValueHolder containing the gradient values.
        """
        gradient = {}

        for name, value in params.items():
            # Create parameter dictionaries for forward and backward evaluation
            params_plus = params.copy()
            params_plus[name] = value + self.step_size / 2

            params_minus = params.copy()
            params_minus[name] = value - self.step_size / 2

            # Calculate forward and backward function values
            f_plus = func(params_plus)
            f_minus = func(params_minus)

            # Central difference derivative
            gradient[name] = (f_plus - f_minus) / self.step_size

        return ValueHolder(gradient)

    def hess(self, func: Callable, params: dict[str, Any]) -> ValueHolder:
        """Compute the Hessian using numerical differentiation.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the Hessian.

        Returns:
            A ValueHolder containing the Hessian values.
        """
        hessian = {}

        # Parameter names
        param_names = list(params.keys())

        for i, name_i in enumerate(param_names):
            hessian[name_i] = {}
            value_i = params[name_i]

            for j, name_j in enumerate(param_names):
                value_j = params[name_j]

                # Diagonal elements
                if i == j:
                    # Create parameter dictionaries
                    params_center = params.copy()

                    params_plus = params.copy()
                    params_plus[name_i] = value_i + self.step_size

                    params_minus = params.copy()
                    params_minus[name_i] = value_i - self.step_size

                    # Calculate function values
                    f_center = func(params_center)
                    f_plus = func(params_plus)
                    f_minus = func(params_minus)

                    # Second derivative (central difference)
                    hessian[name_i][name_j] = (f_plus - 2 * f_center + f_minus) / (
                        self.step_size**2
                    )

                # Off-diagonal elements
                else:
                    # Create parameter dictionaries for mixed partial derivatives
                    params_plus_plus = params.copy()
                    params_plus_plus[name_i] = value_i + self.step_size
                    params_plus_plus[name_j] = value_j + self.step_size

                    params_plus_minus = params.copy()
                    params_plus_minus[name_i] = value_i + self.step_size
                    params_plus_minus[name_j] = value_j - self.step_size

                    params_minus_plus = params.copy()
                    params_minus_plus[name_i] = value_i - self.step_size
                    params_minus_plus[name_j] = value_j + self.step_size

                    params_minus_minus = params.copy()
                    params_minus_minus[name_i] = value_i - self.step_size
                    params_minus_minus[name_j] = value_j - self.step_size

                    # Calculate function values
                    f_plus_plus = func(params_plus_plus)
                    f_plus_minus = func(params_plus_minus)
                    f_minus_plus = func(params_minus_plus)
                    f_minus_minus = func(params_minus_minus)

                    # Mixed partial derivative (central difference)
                    hessian[name_i][name_j] = (
                        f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus
                    ) / (4 * self.step_size**2)

        return ValueHolder(hessian)

    def hvp(
        self, func: Callable, params: dict[str, Any], vector: dict[str, Any]
    ) -> ValueHolder:
        """Compute the Hessian-vector product using numerical differentiation.

        Uses the identity that HVP can be computed as a directional derivative
        of the gradient in the direction of the vector.

        Args:
            func: The function to differentiate.
            params: The parameter values at which to evaluate the HVP.
            vector: The vector to multiply with the Hessian.

        Returns:
            A ValueHolder containing the HVP values.
        """
        # Get base gradient
        grad_base = self.grad(func, params)

        # Create perturbed parameters by moving a small step in the vector direction
        # First, normalize the vector
        vector_norm = znp.sqrt(sum(v**2 for v in vector.values()))
        if vector_norm < 1e-10:  # Avoid division by zero
            return ValueHolder(dict.fromkeys(params, 0.0))

        scale = self.step_size / vector_norm
        params_perturbed = params.copy()
        for name in params:
            if name in vector:
                params_perturbed[name] += vector[name] * scale

        # Get perturbed gradient
        grad_perturbed = self.grad(func, params_perturbed)

        # Compute directional derivative of gradient (HVP)
        hvp_values = {}
        for name in params:
            if name in grad_base and name in grad_perturbed:
                hvp_values[name] = (
                    grad_perturbed[name] - grad_base[name]
                ) / self.step_size
            else:
                hvp_values[name] = 0.0

        return ValueHolder(hvp_values)


def create_differentiator(
    use_jax: bool = True, step_size: float = 1e-6
) -> Differentiator:
    """Create an appropriate differentiator based on the availability of JAX.

    Args:
        use_jax: Whether to use JAX if available.
        step_size: Step size for numerical differentiation (used only if JAX is not available).

    Returns:
        An instance of an appropriate Differentiator subclass.
    """
    if use_jax and _is_jax_available():
        return JaxDifferentiator()
    else:
        return NumericalDifferentiator(step_size=step_size)
