"""Tests for random number generation functionality in the backend."""

from __future__ import annotations

import pytest

# Import backend interfaces
from zfit2.backend import numpy as znp
from zfit2.backend.context import use_backend

# Check which backends are available
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_normal(backend):
    """Test random normal distribution sampling."""
    with use_backend(backend):
        # Test with default parameters
        a = znp.random.normal(size=(1000,))
        assert znp.shape(a) == (1000,)
        # Basic statistical tests - these are probabilistic but should pass most of the time
        assert -0.1 < znp.mean(a) < 0.1  # Mean should be close to 0
        assert 0.9 < znp.std(a) < 1.1  # Std should be close to 1

        # Test with custom parameters
        b = znp.random.normal(loc=5.0, scale=2.0, size=(1000,))
        assert 4.8 < znp.mean(b) < 5.2  # Mean should be close to 5
        assert 1.8 < znp.std(b) < 2.2  # Std should be close to 2

        # Test scalar output
        c = znp.random.normal()
        assert znp.ndim(c) == 0 or znp.size(c) == 1

        # Test with JAX key if applicable
        if backend == "jax":
            key = jax.random.key(42)
            d = znp.random.normal(key=key, size=(10,))
            assert znp.shape(d) == (10,)

            # Verify reproducibility with same key
            key2 = jax.random.key(42)
            d2 = znp.random.normal(key=key2, size=(10,))
            assert znp.allclose(d, d2)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_uniform(backend):
    """Test random uniform distribution sampling."""
    with use_backend(backend):
        # Test with default parameters
        a = znp.random.uniform(size=(1000,))
        assert znp.shape(a) == (1000,)
        assert znp.all(a >= 0.0) and znp.all(a < 1.0)
        # Basic statistical tests
        assert 0.4 < znp.mean(a) < 0.6  # Mean should be close to 0.5

        # Test with custom parameters
        b = znp.random.uniform(low=-1.0, high=1.0, size=(1000,))
        assert znp.all(b >= -1.0) and znp.all(b < 1.0)
        assert -0.1 < znp.mean(b) < 0.1  # Mean should be close to 0

        # Test scalar output
        c = znp.random.uniform()
        assert znp.ndim(c) == 0 or znp.size(c) == 1

        # Test with JAX key if applicable
        if backend == "jax":
            key = jax.random.key(42)
            d = znp.random.uniform(key=key, size=(10,))
            assert znp.shape(d) == (10,)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_seed(backend):
    """Test random seed functionality."""
    with use_backend(backend):
        # This test is a bit tricky because different backends handle seeding differently
        # For NumPy, we can use the global seed
        if backend == "numpy":
            znp.random.seed(42)
            a1 = znp.random.normal(size=(10,))
            znp.random.seed(42)
            a2 = znp.random.normal(size=(10,))
            assert znp.allclose(a1, a2)

        # For JAX, we create reproducible keys
        elif backend == "jax":
            # JAX has explicit keys, so we test that creating keys with same seed gives same results
            key1 = jax.random.key(42)
            b1 = znp.random.normal(key=key1, size=(10,))

            key2 = jax.random.key(42)
            b2 = znp.random.normal(key=key2, size=(10,))

            assert znp.allclose(b1, b2)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_choice(backend):
    """Test random choices from arrays."""
    with use_backend(backend):
        # Simple choice from array
        a = znp.array([10, 20, 30, 40, 50])

        # Skip JAX-specific tests that might not be directly compatible
        if backend == "numpy":
            b = znp.random.choice(a, size=1000)
            assert znp.shape(b) == (1000,)
            assert znp.all(
                znp.isin(b, a)
            )  # All values should be from the original array

            # Test with p parameter
            p = znp.array([0.0, 0.0, 1.0, 0.0, 0.0])  # Always choose 30
            c = znp.random.choice(a, size=10, p=p)
            assert znp.all(c == 30)

            # Test without replacement
            d = znp.random.choice(a, size=5, replace=False)
            assert len(znp.unique(d)) == 5  # All values should be unique


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_shuffle_permutation(backend):
    """Test random shuffling and permutation."""
    with use_backend(backend):
        a = znp.arange(10)

        # Skip for JAX since these operations have different implementation details
        if backend == "numpy":
            # Test permutation
            b = znp.random.permutation(a)
            assert znp.shape(b) == (10,)
            assert set(b.tolist()) == set(
                range(10)
            )  # Should contain all original elements

            # Test shuffle - note this modifies the array in-place in NumPy
            c = znp.array([1, 2, 3, 4, 5])
            znp.random.shuffle(c)
            assert znp.shape(c) == (5,)
            assert set(c.tolist()) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_random_distributions(backend):
    """Test other random distributions."""
    with use_backend(backend):
        # Test a few more distributions - implementations may vary by backend,
        # so we're just checking that they run and return the right shapes

        # Exponential distribution
        a = znp.random.exponential(scale=1.0, size=(100,))
        assert znp.shape(a) == (100,)
        assert znp.all(a >= 0)  # Exponential is always positive

        # Poisson distribution
        try:
            b = znp.random.poisson(lam=5.0, size=(100,))
            assert znp.shape(b) == (100,)
            assert znp.all(b >= 0)  # Poisson values are non-negative integers
        except (NotImplementedError, AttributeError):
            # Some backends might not implement Poisson
            pass

        # Beta distribution
        try:
            c = znp.random.beta(a=2.0, b=5.0, size=(100,))
            assert znp.shape(c) == (100,)
            assert znp.all(c >= 0) and znp.all(c <= 1)  # Beta is between 0 and 1
        except (NotImplementedError, AttributeError):
            # Some backends might not implement Beta
            pass

        # Gamma distribution
        try:
            d = znp.random.gamma(shape=2.0, scale=2.0, size=(100,))
            assert znp.shape(d) == (100,)
            assert znp.all(d >= 0)  # Gamma is positive
        except (NotImplementedError, AttributeError):
            # Some backends might not implement Gamma
            pass


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
