"""Log-likelihood module for zfit2.

This module provides classes for negative log-likelihood calculation
and optimization, with support for various summation methods and
numerical stability improvements for handling floating-point issues.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import jax.lax

from .backend import numpy as znp
from .diff import create_differentiator
from .dist import Distribution
from .parameter import Parameter, Parameters
from .statistic import Statistics
from .valueholder import ValueHolder


class SumMethod:
    """Enumeration of available summation methods for log-likelihoods."""
    DIRECT = "direct"        # Simple sum
    STABLE = "stable"        # Sorted sum (more stable)
    PAIRWISE = "pairwise"    # Divide-and-conquer summation
    KAHAN = "kahan"          # Kahan compensated summation
    NEUMAIER = "neumaier"    # Neumaier improved compensated summation
    LOGSUMEXP = "logsumexp"  # Log-sum-exp for numerical stability


class ZeroHandler:
    """Methods for handling zero values in log-likelihood calculations."""
    IGNORE = "ignore"        # Ignore zeros in the computation (skip them)
    EPSILON = "epsilon"      # Add a small epsilon to zeros
    CLAMP = "clamp"          # Clamp values to a minimum threshold
    SMOOTHED = "smoothed"    # Use a smoothed transition function
    SHIFTED = "shifted"      # Shift all values by the absolute minimum


class NLL(Statistics):
    """Negative Log-Likelihood for a distribution.

    This class calculates the negative log-likelihood for a distribution given data.
    It provides methods for calculating pointwise log-likelihoods, summation of log-likelihoods,
    and gradient and Hessian calculations.
    """

    def __init__(
            self,
            distribution: Distribution,
            data: Mapping[str, Union[float, np.ndarray]],
            *,
            name: Optional[str] = None,
            use_jax: bool = True,
            constant_subtraction: Union[str, float, None] = None,
            zero_handling: str = ZeroHandler.EPSILON,
            epsilon: float = 1e-10,
            min_value: float = -1e10,
    ):
        """Initialize the Negative Log-Likelihood.

        Args:
            distribution: The distribution to calculate the log-likelihood for.
            data: The data to calculate the log-likelihood with, as a mapping from variable names to values.
            name: Optional name for the likelihood.
            use_jax: Whether to use JAX for gradient calculations.
            constant_subtraction: Strategy for constant subtraction:
                - None: No subtraction
                - float: Specific constant value to subtract
                - "auto": Estimate constant from first calculation
                - "per_point": Subtract a constant for each data point
            zero_handling: Method to handle zero or negative values in log-likelihoods:
                - "ignore": Skip zero values in summation
                - "epsilon": Add a small epsilon value
                - "clamp": Clamp values to a minimum threshold
                - "smoothed": Use a smoothly interpolated function
                - "shifted": Shift all values by the absolute minimum
            epsilon: Small value for numerical stability (used with "epsilon" handling).
            min_value: Minimum value for log-likelihoods (used with "clamp" handling).
        """
        super().__init__(name=name or f"NLL_{distribution.name}")
        self.distribution = distribution
        if isinstance(data, dict):
            data = ValueHolder(data)
        self.data = data
        self.differentiator = create_differentiator(use_jax=use_jax)

        # Set up constant subtraction
        self.constant_subtraction = constant_subtraction

        # Set up zero handling
        if zero_handling not in vars(ZeroHandler).values():
            valid_methods = [m for m in dir(ZeroHandler) if not m.startswith('_')]
            msg = f"Invalid zero handling method: {zero_handling}. Valid methods: {valid_methods}"
            raise ValueError(msg)
        self.zero_handling = zero_handling
        self.epsilon = epsilon
        self.min_value = min_value

        # Initialize constants for constant subtraction
        self.constant = None
        self.point_constants = None

        # Calculate initial values for constant subtraction if needed
        if constant_subtraction == "auto" or constant_subtraction == "per_point":
            # Calculate pointwise log-likelihood once
            pointwise = self.pointwise_loglik()

            # For per-point subtraction, store the pointwise values
            if constant_subtraction == "per_point":
                self.point_constants = pointwise

            # For auto subtraction, calculate and store the sum
            if constant_subtraction == "auto":
                # Use direct sum for initialization
                self.constant = znp.sum(pointwise)

    def pointwise_loglik(
            self, params: Optional[Mapping[str, Union[float, np.ndarray]]] = None
    ) -> np.ndarray:
        """Calculate the pointwise log-likelihood for each data point.

        Args:
            params: Optional mapping of parameter names to values. If not provided,
                   the current values of the distribution parameters are used.

        Returns:
            An array of log-likelihoods, one for each data point.
        """
        # Extract data values for the domain variables
        data_values = {var.name: self.data[var.name] for var in self.distribution.domain.variables}

        # Call the log_pdf method of the distribution
        loglik_values = self.distribution.log_pdf(data_values, params=params)

        # Handle numerical issues with zeros and negative values
        loglik_values = self._handle_zeros(loglik_values)

        return loglik_values

    def _handle_zeros(self, values: np.ndarray) -> np.ndarray:
        """Handle zero or negative values in log-likelihoods according to the selected strategy.

        Args:
            values: Array of log-likelihood values.

        Returns:
            Modified array with handled zero/negative values.
        """
        if self.zero_handling == ZeroHandler.IGNORE:
            # Values will be ignored during summation, no change needed here
            return values

        elif self.zero_handling == ZeroHandler.EPSILON:
            # Add a small epsilon to avoid log(0)
            # For very negative values, exp(value) will be close to zero
            # log(exp(value) + epsilon) ≈ log(epsilon) for very negative values
            return znp.log(znp.exp(values) + self.epsilon)

        elif self.zero_handling == ZeroHandler.CLAMP:
            # Clamp values to a minimum threshold
            return znp.maximum(values, self.min_value)

        elif self.zero_handling == ZeroHandler.SMOOTHED:
            # Use a smoothed transition function
            # For x > threshold: f(x) = x
            # For x <= threshold: f(x) smoothly approaches min_value
            threshold = self.min_value / 2
            above_threshold = values > threshold

            # For values below threshold, use smooth transition
            below = ~above_threshold
            if znp.any(below):
                # Create a mask for the below values
                mask = znp.zeros_like(values, dtype=bool)
                mask = mask.at[znp.where(below)[0]].set(True)

                # Calculate the smoothed values for all elements
                # (will only be used where mask is True)
                alpha = 1.0  # Controls smoothness of transition
                smoothed = self.min_value + (threshold - self.min_value) * znp.exp(
                    alpha * (values - threshold) / (threshold - self.min_value)
                )

                # Use where to create a new array without in-place modification
                return znp.where(mask, smoothed, values)

            # If no values are below threshold, return the original values
            return values

        elif self.zero_handling == ZeroHandler.SHIFTED:
            # Shift all values by the absolute minimum
            min_val = znp.min(values)
            if min_val < 0:
                return values - min_val + self.epsilon
            return values

        # Default case - return values unchanged
        return values

    def sum_loglik(
            self,
            params: Optional[Mapping[str, Union[float, np.ndarray]]] = None,
            method: str = SumMethod.DIRECT
    ) -> Union[float, np.ndarray]:
        """Calculate the sum of log-likelihoods using the specified method.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation. Options from SumMethod.

        Returns:
            The total log-likelihood.
        """
        pointwise = self.pointwise_loglik(params)

        # Apply per-point constant subtraction if enabled
        if self.constant_subtraction == "per_point":
            # Use the pre-calculated point_constants
            pointwise = pointwise - self.point_constants

        # Special handling for ZeroHandler.IGNORE case
        if self.zero_handling == ZeroHandler.IGNORE:
            # Filter out -inf or very negative values before summation
            valid_mask = znp.isfinite(pointwise) & (pointwise > self.min_value)
            if not znp.any(valid_mask):
                return self.min_value  # All values are invalid
            pointwise = pointwise[valid_mask]

        # Choose summation method
        if method == SumMethod.DIRECT:
            result = znp.sum(pointwise)
        elif method == SumMethod.STABLE:
            result = self._stable_sum(pointwise)
        elif method == SumMethod.PAIRWISE:
            result = self._pairwise_sum(pointwise)
        elif method == SumMethod.KAHAN:
            result = self._kahan_sum(pointwise)
        elif method == SumMethod.NEUMAIER:
            result = self._neumaier_sum(pointwise)
        elif method == SumMethod.LOGSUMEXP:
            result = self._logsumexp_sum(pointwise)
        else:
            msg = f"Unknown summation method: {method}"
            raise ValueError(msg)

        # Apply global constant subtraction if enabled
        if isinstance(self.constant_subtraction, (int, float)):
            result = result - self.constant_subtraction
        elif self.constant_subtraction == "auto":
            # Use the pre-calculated constant
            result = result - self.constant

        return result

    def nll(
            self,
            params: Optional[Mapping[str, Union[float, np.ndarray]]] = None,
            method: str = SumMethod.DIRECT
    ) -> Union[float, np.ndarray]:
        """Calculate the negative log-likelihood.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            The negative log-likelihood value.
        """
        # Return the JAX array directly without converting to float
        return -self.sum_loglik(params, method=method)

    def calculate(self, params: Optional[Mapping[str, Any]] = None) -> Union[float, np.ndarray]:
        """Calculate the negative log-likelihood (Statistics interface method).

        Args:
            params: Optional mapping of parameter names to values.

        Returns:
            The negative log-likelihood value.
        """
        return self.nll(params)

    def grad(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate the gradient of the negative log-likelihood.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A ValueHolder containing the gradient values.
        """
        params_dict = self._prepare_params(params)

        # Define a function that takes parameter values and returns the NLL
        def nll_func(p):
            return self.nll(p, method=method)

        return self.differentiator.grad(nll_func, params_dict)

    def hess(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate the Hessian of the negative log-likelihood.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A ValueHolder containing the Hessian values.
        """
        params_dict = self._prepare_params(params)

        # Define a function that takes parameter values and returns the NLL
        def nll_func(p):
            return self.nll(p, method=method)

        return self.differentiator.hess(nll_func, params_dict)

    def hvp(
            self,
            vector: Mapping[str, Any],
            params: Optional[Mapping[str, Any]] = None,
            method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate Hessian-vector product (more efficient than full Hessian).

        Args:
            vector: The vector to multiply with the Hessian.
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A ValueHolder containing the Hessian-vector product.
        """
        params_dict = self._prepare_params(params)

        # Define a function that takes parameter values and returns the NLL
        def nll_func(p):
            return self.nll(p, method=method)

        return self.differentiator.hvp(nll_func, params_dict, vector)

    def val_and_grad(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> Tuple[Union[float, np.ndarray], ValueHolder]:
        """Calculate both the NLL value and its gradient.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A tuple of (NLL value, gradient).
        """
        params_dict = self._prepare_params(params)

        # Define a function that takes parameter values and returns the NLL
        def nll_func(p):
            return self.nll(p, method=method)

        value, gradient = self.differentiator.value_and_grad(nll_func, params_dict)
        # Return the JAX array directly without converting to float
        return value, gradient

    def _prepare_params(self, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Prepare parameters for calculation.

        Args:
            params: Optional mapping of parameter names to values.

        Returns:
            A dictionary of parameter names to values.
        """
        if params is None:
            # Use the current values of the distribution parameters
            return {param.name: param.value for param in self.distribution.params.params}
        elif isinstance(params, ValueHolder):
            return params._values
        else:
            return dict(params)

    def __add__(self, other: Union[NLL, 'NLLS']) -> 'NLLS':
        """Add this NLL to another NLL or NLLS.

        Args:
            other: Another NLL or NLLS object.

        Returns:
            A new NLLS object that combines both likelihoods.
        """
        if isinstance(other, NLL):
            return NLLS([self, other])
        elif isinstance(other, NLLS):
            return NLLS([self] + other.likelihoods)
        else:
            msg = f"Cannot add NLL to object of type {type(other)}"
            raise TypeError(msg)

    # Summation methods
    def _stable_sum(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate a numerically stable sum by sorting values."""
        sorted_values = znp.sort(values)
        return znp.sum(sorted_values)

    def _pairwise_sum(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate sum using pairwise summation for better numerical stability.

        Pairwise summation is a divide-and-conquer approach with O(log n) error growth.
        """
        # Implement pairwise summation without modifying the input array
        def pairwise_sum_recursive(arr):
            n = len(arr)
            if n == 0:
                return 0
            if n == 1:
                return arr[0]

            # If odd length, handle the last element separately
            if n % 2 == 1:
                return pairwise_sum_recursive(arr[:-1]) + arr[-1]

            # Sum pairs and recurse
            pairs = arr[0::2] + arr[1::2]
            return pairwise_sum_recursive(pairs)

        return pairwise_sum_recursive(values)

    def _kahan_sum(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate sum using Kahan summation algorithm for error compensation."""
        # JAX-friendly implementation using scan
        def kahan_step(carry, val):
            sum_val, compensation = carry
            # Apply compensation from previous iteration
            corrected = val - compensation
            # Add to sum (some low-order bits will be lost)
            temp_sum = sum_val + corrected
            # Compute the compensation (the lost low-order bits)
            new_compensation = (temp_sum - sum_val) - corrected
            return (temp_sum, new_compensation), None

        # Initial state: (sum, compensation)
        initial = (znp.array(0.0), znp.array(0.0))

        # Use JAX's scan for a functional implementation
        (sum_val, _), _ = jax.lax.scan(kahan_step, initial, values)

        return sum_val

    def _neumaier_sum(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate sum using Neumaier's algorithm (improved Kahan).

        This algorithm is more robust when adding values with large magnitude differences.
        Implemented using JAX's scan for functional programming compatibility.
        """
        # Define a step function for Neumaier summation
        def neumaier_step(carry, val):
            sum_val, compensation = carry

            # Add value to sum
            temp_sum = sum_val + val

            # Calculate compensation term
            # If sum_val is larger, low-order bits of val are lost
            # If val is larger, low-order bits of sum_val are lost
            new_compensation = znp.where(
                znp.abs(sum_val) >= znp.abs(val),
                compensation + ((sum_val - temp_sum) + val),
                compensation + ((val - temp_sum) + sum_val)
            )

            return (temp_sum, new_compensation), None

        # Initial state: (sum, compensation)
        initial = (znp.array(0.0), znp.array(0.0))

        # Use JAX's scan for a functional implementation
        (sum_val, compensation), _ = jax.lax.scan(neumaier_step, initial, values)

        # Add the compensation to the final sum
        return sum_val + compensation

    def _logsumexp_sum(self, values: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate log(sum(exp(values))) in a numerically stable way.

        This implementation follows the shifted log-sum-exp approach:
        log(sum(exp(x_i))) = a + log(sum(exp(x_i - a)))

        where a = max(x_i) to prevent overflow.

        Args:
            values: Array of log values to sum.

        Returns:
            Calculated log-sum-exp value.
        """
        if len(values) == 0:
            return -znp.inf

        max_val = znp.max(values)
        if znp.isinf(max_val) and max_val < 0:
            return -znp.inf  # All values are -inf

        # Shift values by max to avoid overflow
        shifted = values - max_val

        # Calculate sum(exp(shifted)) and take log
        # Using built-in logsumexp when available for better precision
        if self.differentiator.use_jax:
            from jax.nn import logsumexp
            return logsumexp(values)
        else:
            try:
                from scipy.special import logsumexp
                return logsumexp(values)
            except ImportError:
                # Manual implementation if scipy not available
                return max_val + znp.log(znp.sum(znp.exp(shifted)))


class NLLS(Statistics):
    """A collection of Negative Log-Likelihoods.

    This class combines multiple NLL objects to form a joint likelihood
    that can be minimized together.
    """

    def __init__(self, likelihoods: Sequence[NLL], *, name: Optional[str] = None):
        """Initialize the collection of Negative Log-Likelihoods.

        Args:
            likelihoods: A sequence of NLL objects.
            name: Optional name for the likelihood collection.
        """
        super().__init__(name=name or "NLLS")
        self.likelihoods = list(likelihoods)

    def nll(self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT) -> float:
        """Calculate the sum of negative log-likelihoods.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation (passed to each NLL).

        Returns:
            The sum of negative log-likelihoods.
        """
        return sum(nll.nll(params, method=method) for nll in self.likelihoods)

    def calculate(self, params: Optional[Mapping[str, Any]] = None) -> float:
        """Calculate the sum of negative log-likelihoods (Statistics interface method).

        Args:
            params: Optional mapping of parameter names to values.

        Returns:
            The sum of negative log-likelihoods.
        """
        return self.nll(params)

    def pointwise_loglik(
            self, params: Optional[Mapping[str, Union[float, np.ndarray]]] = None
    ) -> List[np.ndarray]:
        """Calculate the pointwise log-likelihood for each NLL object.

        Args:
            params: Optional mapping of parameter names to values.

        Returns:
            A list of arrays, each containing the pointwise log-likelihood for one NLL.
        """
        return [nll.pointwise_loglik(params) for nll in self.likelihoods]

    def sum_loglik(
            self, params: Optional[Mapping[str, Union[float, np.ndarray]]] = None,
            method: str = SumMethod.DIRECT
    ) -> float:
        """Calculate the sum of log-likelihoods across all NLL objects.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation (passed to each NLL).

        Returns:
            The total log-likelihood.
        """
        return sum(nll.sum_loglik(params, method=method) for nll in self.likelihoods)

    def grad(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate the gradient of the sum of negative log-likelihoods.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation (passed to each NLL).

        Returns:
            A ValueHolder containing the combined gradient values.
        """
        # Calculate gradients for each likelihood
        gradients = [nll.grad(params, method=method) for nll in self.likelihoods]

        # Get all parameter names
        param_names = set()
        for grad in gradients:
            param_names.update(grad._values.keys())

        # Sum up the gradients
        combined_gradient = {}
        for name in param_names:
            combined_gradient[name] = sum(
                grad._values.get(name, 0) for grad in gradients
            )

        return ValueHolder(combined_gradient)

    def hess(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate the Hessian of the sum of negative log-likelihoods.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation (passed to each NLL).

        Returns:
            A ValueHolder containing the combined Hessian values.
        """
        # Calculate Hessians for each likelihood
        hessians = [nll.hess(params, method=method) for nll in self.likelihoods]

        # Get all parameter names
        param_names = set()
        for hess in hessians:
            param_names.update(hess._values.keys())

        # Initialize the combined Hessian
        combined_hessian = {}
        for name_i in param_names:
            combined_hessian[name_i] = {}
            for name_j in param_names:
                combined_hessian[name_i][name_j] = 0

        # Sum up the Hessians
        for hess in hessians:
            for name_i, row in hess._values.items():
                for name_j, value in row.items():
                    combined_hessian[name_i][name_j] += value

        return ValueHolder(combined_hessian)

    def hvp(
            self,
            vector: Mapping[str, Any],
            params: Optional[Mapping[str, Any]] = None,
            method: str = SumMethod.DIRECT
    ) -> ValueHolder:
        """Calculate Hessian-vector product for the combined NLLs.

        Args:
            vector: The vector to multiply with the Hessian.
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A ValueHolder containing the combined Hessian-vector product.
        """
        # Calculate HVPs for each likelihood
        hvps = [nll.hvp(vector, params, method=method) for nll in self.likelihoods]

        # Get all parameter names
        param_names = set()
        for hvp in hvps:
            param_names.update(hvp._values.keys())

        # Sum up the HVPs
        combined_hvp = {}
        for name in param_names:
            combined_hvp[name] = sum(
                hvp._values.get(name, 0) for hvp in hvps
            )

        return ValueHolder(combined_hvp)

    def val_and_grad(
            self, params: Optional[Mapping[str, Any]] = None, method: str = SumMethod.DIRECT
    ) -> Tuple[Union[float, np.ndarray], ValueHolder]:
        """Calculate both the NLL value and its gradient.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation.

        Returns:
            A tuple of (NLL value, gradient).
        """
        value = self.nll(params, method=method)
        gradient = self.grad(params, method=method)
        # Return the JAX array directly without converting to float
        return value, gradient

    def __add__(self, other: Union[NLL, 'NLLS']) -> 'NLLS':
        """Add this NLLS to another NLL or NLLS.

        Args:
            other: Another NLL or NLLS object.

        Returns:
            A new NLLS object that combines both likelihoods.
        """
        if isinstance(other, NLL):
            return NLLS(self.likelihoods + [other])
        elif isinstance(other, NLLS):
            return NLLS(self.likelihoods + other.likelihoods)
        else:
            msg = f"Cannot add NLLS to object of type {type(other)}"
            raise TypeError(msg)


class OptimizedNLL(NLL):
    """An optimized version of the Negative Log-Likelihood with robustness transformations.

    This class extends NLL with additional numerical stability improvements:
    - Constant subtraction (global or per-point)
    - Advanced zero/negative value handling
    - Better default summation method
    """

    def __init__(
            self,
            distribution: Distribution,
            data: Mapping[str, Union[float, np.ndarray]],
            *,
            name: Optional[str] = None,
            use_jax: bool = True,
            constant_subtraction: str = "auto",
            zero_handling: str = ZeroHandler.SMOOTHED,
            epsilon: float = 1e-10,
            min_value: float = -1e10,
            sum_method: str = SumMethod.NEUMAIER,
    ):
        """Initialize the Optimized Negative Log-Likelihood.

        Args:
            distribution: The distribution to calculate the likelihood for.
            data: The data to calculate the likelihood with, as a mapping from variable names to values.
            name: Optional name for the likelihood.
            use_jax: Whether to use JAX for gradient calculations.
            constant_subtraction: Strategy for constant subtraction:
                - "auto": Estimate constant from first calculation (default)
                - "per_point": Subtract a constant for each data point
            zero_handling: Method to handle zero or negative values (default: "smoothed")
            epsilon: Small value for numerical stability.
            min_value: Minimum value for log-likelihoods in clamping.
            sum_method: Default summation method to use.
        """
        super().__init__(
            distribution=distribution,
            data=data,
            name=name,
            use_jax=use_jax,
            constant_subtraction=constant_subtraction,
            zero_handling=zero_handling,
            epsilon=epsilon,
            min_value=min_value,
        )
        self.sum_method = sum_method

    def nll(
            self,
            params: Optional[Mapping[str, Union[float, np.ndarray]]] = None,
            method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """Calculate the optimized negative log-likelihood.

        Args:
            params: Optional mapping of parameter names to values.
            method: Method to use for summation. If None, uses the default method.

        Returns:
            The negative log-likelihood value with optimizations.
        """
        if method is None:
            method = self.sum_method

        # Return the JAX array directly without converting to float
        return -self.sum_loglik(params, method=method)
