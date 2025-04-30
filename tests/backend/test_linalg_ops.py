"""Tests for linear algebra functionality in the backend."""

from __future__ import annotations

import numpy as np
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
def test_matrix_operations(backend):
    """Test basic matrix operations."""
    with use_backend(backend):
        # Create test matrices
        a = znp.array([[1.0, 2.0], [3.0, 4.0]])
        b = znp.array([[5.0, 6.0], [7.0, 8.0]])

        # Test matrix multiplication
        c = znp.matmul(a, b)
        expected_c = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert znp.allclose(c, expected_c)

        # Test @ operator (if available)
        try:
            d = a @ b
            assert znp.allclose(d, expected_c)
        except (TypeError, AttributeError):
            # Some backends might not support @ operator
            pass

        # Test dot product
        v1 = znp.array([1.0, 2.0, 3.0])
        v2 = znp.array([4.0, 5.0, 6.0])
        dot_product = znp.dot(v1, v2)
        assert znp.isclose(dot_product, 32.0)

        # Test matrix-vector multiplication
        mv = znp.matmul(a, znp.array([1.0, 2.0]))
        assert znp.allclose(mv, np.array([5.0, 11.0]))


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_linalg_inv(backend):
    """Test matrix inversion."""
    with use_backend(backend):
        # Create an invertible matrix
        a = znp.array([[4.0, 7.0], [2.0, 6.0]])

        # Compute inverse
        a_inv = znp.linalg.inv(a)

        # Check that A * A^-1 ≈ I
        identity = znp.matmul(a, a_inv)
        expected_identity = znp.eye(2)
        assert znp.allclose(identity, expected_identity, atol=1e-5)

        # Check specific values of the inverse
        expected_inv = np.array([[0.6, -0.7], [-0.2, 0.4]])
        assert znp.allclose(a_inv, expected_inv, atol=1e-5)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_linalg_solve(backend):
    """Test solving linear systems."""
    with use_backend(backend):
        # Create a system of equations: Ax = b
        a = znp.array([[3.0, 1.0], [1.0, 2.0]])
        b = znp.array([9.0, 8.0])

        # Solve the system
        x = znp.linalg.solve(a, b)

        # Check the solution
        assert znp.allclose(znp.matmul(a, x), b)
        expected_x = np.array([2.0, 3.0])
        assert znp.allclose(x, expected_x)

        # Test with multiple right-hand sides
        b_multiple = znp.array([[9.0, 18.0], [8.0, 16.0]])
        x_multiple = znp.linalg.solve(a, b_multiple)
        assert znp.allclose(znp.matmul(a, x_multiple), b_multiple)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_linalg_eigendecomposition(backend):
    """Test eigenvalue decomposition."""
    with use_backend(backend):
        # Create a symmetric matrix
        a = znp.array([[1.0, 2.0], [2.0, 4.0]])

        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = znp.linalg.eigh(a)

        # Check that eigenvalues are sorted (smallest to largest)
        assert eigenvals[0] <= eigenvals[1]

        # Check that Av = λv for each eigenpair
        for i in range(len(eigenvals)):
            v = eigenvecs[:, i]
            lam = eigenvals[i]
            assert znp.allclose(znp.matmul(a, v), lam * v, atol=1e-5)

        # Check expected eigenvalues (approximately 0 and 5)
        # The eigenvectors can vary, so we don't check them directly
        expected_eigenvals = np.array([0.0, 5.0])
        assert znp.allclose(znp.sort(eigenvals), expected_eigenvals, atol=1e-5)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_linalg_cholesky(backend):
    """Test Cholesky decomposition."""
    with use_backend(backend):
        # Create a positive definite matrix
        a = znp.array([[4.0, 2.0], [2.0, 5.0]])

        # Compute Cholesky decomposition
        l = znp.linalg.cholesky(a)

        # Check that L is lower triangular
        assert znp.isclose(l[0, 1], 0.0)

        # Check that L * L^T = A
        l_transpose = znp.transpose(l)
        reconstructed = znp.matmul(l, l_transpose)
        assert znp.allclose(reconstructed, a)

        # Check specific values of L
        expected_l = np.array([[2.0, 0.0], [1.0, 2.0]])
        assert znp.allclose(l, expected_l)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_tensordot(backend):
    """Test tensor dot product."""
    with use_backend(backend):
        # Create tensors
        a = znp.arange(6).reshape(2, 3)
        b = znp.arange(12).reshape(3, 4)

        # Test tensordot with axes=1 (contract last dim of a with first dim of b)
        c = znp.tensordot(a, b, axes=1)
        expected_c = np.array([[20, 23, 26, 29], [56, 68, 80, 92]])
        assert znp.allclose(c, expected_c)

        # Test tensordot with explicit axes
        d = znp.tensordot(a, b, axes=([1], [0]))
        expected_d = np.array([[20, 23, 26, 29], [56, 68, 80, 92]])
        assert znp.allclose(d, expected_d)


@pytest.mark.parametrize("backend", ["numpy"] + (["jax"] if HAS_JAX else []))
def test_advanced_linalg(backend):
    """Test more advanced linear algebra operations."""
    with use_backend(backend):
        # Create test matrices
        a = znp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

        # Test matrix determinant - won't test the exact function since it might
        # differ between backends, but we'll check that results match numpy
        try:
            # Implementations can vary, so we're just checking it runs
            det_a = znp.linalg.det(a)
            assert znp.ndim(det_a) == 0  # Should be a scalar

            np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
            np_det = np.linalg.det(np_a)

            assert znp.isclose(det_a, np_det)
        except (NotImplementedError, AttributeError):
            # Some backends might not implement det
            pass

        # Test matrix rank
        try:
            rank_a = znp.linalg.matrix_rank(a)
            assert rank_a == 3  # This matrix has full rank

            # Create a rank-deficient matrix
            b = znp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
            rank_b = znp.linalg.matrix_rank(b)
            assert rank_b == 1  # This matrix has rank 1
        except (NotImplementedError, AttributeError):
            # Some backends might not implement matrix_rank
            pass

        # Test norm
        try:
            # Vector norm
            v = znp.array([3.0, 4.0])
            norm_v = znp.linalg.norm(v)
            assert znp.isclose(norm_v, 5.0)  # 3-4-5 triangle

            # Matrix norm
            norm_a = znp.linalg.norm(a)
            np_norm = np.linalg.norm(np_a)
            assert znp.isclose(norm_a, np_norm)
        except (NotImplementedError, AttributeError):
            # Some backends might not implement norm
            pass


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
