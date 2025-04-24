import jax
from jax import numpy as jnp
from jax import random

class Statistics:
    """
    Represents mathematically a statistics, that is, a function of a random variable.
    """

    def __init__(self, name):
        """
        Initializes the Statistics object.

        Args:
            name (str): The name of the statistics.
        """
        self.name = name

    def calculate(self, data):
        """
        Calculates the statistics based on the provided data.

        Args:
            data (list or numpy.ndarray): The data to perform the calculation on.

        Returns:
            float: The calculated statistics value.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def __str__(self):
        return f"Statistics: {self.name}"


class LogLikelihood(Statistics):
    def __init__(self, name):
        super().__init__(name)

    def calculate(self, data):
        """
        Calculates the log-likelihood statistic.

        Args:
            data (list or numpy.ndarray): The data to perform the calculation on.

        Returns:
            float: The calculated log-likelihood value.
        """
        if len(data) == 0:
            return 0.0
        return jnp.sum(jnp.log(data))


class Chi2(Statistics):
    def __init__(self, name):
        super().__init__(name)

    def calculate(self, data, mean, variance):
        """
        Calculates the chi-squared statistic.

        Args:
            data (list or numpy.ndarray): The data to perform the calculation on.
            mean (float): The mean of the distribution.
            variance (float): The variance of the distribution.

        Returns:
            float: The calculated chi-squared value.
        """
        if variance <= 0:
            return 0.0  # Avoid division by zero
        return jnp.sum(((data - mean)**2) / variance)


class Mean(Statistics):
    def __init__(self, name):
        super().__init__(name)

    def calculate(self, data):
        return jnp.mean(data)


class OptimizedStatistics(Statistics):
    def __init__(self, name):
        super().__init__(name)

    def calculate(self, data):
        raise NotImplementedError("Subclasses must implement calculate method")

    def gradient(self, data):
        """
        Calculates the gradient of the statistics with respect to the data.
        """
        raise NotImplementedError("Subclasses must implement gradient method")

    def hessian(self, data):
        """
        Calculates the Hessian matrix of the statistics.
        """
        raise NotImplementedError("Subclasses must implement hessian method")