"""NumPy-like interface to the zfit2 backend system.

This module provides a NumPy-compatible interface to the active backend,
allowing code to be written using familiar NumPy syntax while leveraging
the computational capabilities of the active backend (JAX, NumPy, or SymPy).

Example:
    >>> import zfit2.backend.numpy as znp
    >>> x = znp.array([1, 2, 3])
    >>> y = znp.sin(x) + znp.cos(x)
    >>> znp.random.normal(size=(3, 3))
"""

from __future__ import annotations

from typing import Any

from . import get_backend

# This module dynamically forwards calls to the active backend
# We define stub functions and classes that proxy to the active backend

# ============================
# Core Array Creation Functions
# ============================


def array(obj, dtype=None, copy=None, device=None) -> Any:
    """Create an array."""
    return get_backend().array(obj, dtype=dtype, copy=copy, device=device)


def asarray(a, dtype=None, copy=None, device=None) -> Any:
    """Convert the input to an array."""
    return get_backend().asarray(a, dtype=dtype, copy=copy, device=device)


def zeros(shape, dtype=None, device=None) -> Any:
    """Return a new array of given shape and type, filled with zeros."""
    return get_backend().zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None) -> Any:
    """Return a new array of given shape and type, filled with ones."""
    return get_backend().ones(shape, dtype=dtype, device=device)


def full(shape, fill_value, dtype=None, device=None) -> Any:
    """Return a new array of given shape and type, filled with fill_value."""
    return get_backend().full(shape, fill_value, dtype=dtype, device=device)


def zeros_like(a, dtype=None, device=None) -> Any:
    """Return an array of zeros with the same shape and type as a given array."""
    shape = get_backend().shape(a)
    return get_backend().zeros(
        shape, dtype=(dtype or get_backend().dtype(a)), device=device
    )


def ones_like(a, dtype=None, device=None) -> Any:
    """Return an array of ones with the same shape and type as a given array."""
    shape = get_backend().shape(a)
    return get_backend().ones(
        shape, dtype=(dtype or get_backend().dtype(a)), device=device
    )


def full_like(a, fill_value, dtype=None, device=None) -> Any:
    """Return an array of fill_value with the same shape and type as a given array."""
    shape = get_backend().shape(a)
    return get_backend().full(
        shape, fill_value, dtype=(dtype or get_backend().dtype(a)), device=device
    )


def arange(start, stop=None, step=1, dtype=None, device=None) -> Any:
    """Return evenly spaced values within a given interval."""
    return get_backend().arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, dtype=None, device=None) -> Any:
    """Return evenly spaced numbers over a specified interval."""
    return get_backend().linspace(start, stop, num, endpoint=endpoint, dtype=dtype)


def eye(N, M=None, k=0, dtype=None, device=None) -> Any:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
    return get_backend().eye(N, M=M, k=k, dtype=dtype)


def identity(n, dtype=None, device=None) -> Any:
    """Return the identity array of size n."""
    return get_backend().identity(n, dtype=dtype)


def empty(shape, dtype=None, device=None) -> Any:
    """Return a new array of given shape and type, without initializing entries."""
    return get_backend().empty(shape, dtype=dtype)


def empty_like(a, dtype=None, device=None) -> Any:
    """Return an empty array with the same shape and type as a given array."""
    shape = get_backend().shape(a)
    return get_backend().empty(
        shape, dtype=(dtype or get_backend().dtype(a)), device=device
    )


# ============================
# Array Manipulation Functions
# ============================


def reshape(a, newshape) -> Any:
    """Gives a new shape to an array without changing its data."""
    return get_backend().reshape(a, newshape)


def transpose(a, axes=None) -> Any:
    """Permute the dimensions of an array."""
    return get_backend().transpose(a, axes=axes)


def concatenate(arrays, axis=0) -> Any:
    """Join a sequence of arrays along an existing axis."""
    return get_backend().concatenate(arrays, axis=axis)


def stack(arrays, axis=0) -> Any:
    """Join a sequence of arrays along a new axis."""
    return get_backend().stack(arrays, axis=axis)


def vstack(tup) -> Any:
    """Stack arrays in sequence vertically (row wise)."""
    return get_backend().vstack(tup)


def hstack(tup) -> Any:
    """Stack arrays in sequence horizontally (column wise)."""
    return get_backend().hstack(tup)


def split(ary, indices_or_sections, axis=0) -> Any:
    """Split an array into multiple sub-arrays."""
    return get_backend().split(ary, indices_or_sections, axis=axis)


def tile(A, reps) -> Any:
    """Construct an array by repeating A the number of times given by reps."""
    return get_backend().tile(A, reps)


def repeat(a, repeats, axis=None) -> Any:
    """Repeat elements of an array."""
    return get_backend().repeat(a, repeats, axis=axis)


def expand_dims(a, axis) -> Any:
    """Expand the shape of an array."""
    return get_backend().expand_dims(a, axis)


def squeeze(a, axis=None) -> Any:
    """Remove single-dimensional entries from the shape of an array."""
    return get_backend().squeeze(a, axis=axis)


def swapaxes(a, axis1, axis2) -> Any:
    """Interchange two axes of an array."""
    return get_backend().swapaxes(a, axis1, axis2)


def moveaxis(a, source, destination) -> Any:
    """Move axes of an array to new positions."""
    return get_backend().moveaxis(a, source, destination)


def where(condition, x=None, y=None) -> Any:
    """Return elements chosen from x or y depending on condition."""
    return get_backend().where(condition, x, y)


def pad(array, pad_width, mode="constant", **kwargs) -> Any:
    """Pad an array."""
    return get_backend().pad(array, pad_width, mode=mode, **kwargs)


def meshgrid(*xi, indexing="xy") -> Any:
    """Return coordinate matrices from coordinate vectors."""
    return get_backend().meshgrid(*xi, indexing=indexing)


# ============================
# Mathematical Functions
# ============================


def sum(a, axis=None, dtype=None, keepdims=False) -> Any:
    """Sum of array elements over a given axis."""
    return get_backend().sum(a, axis=axis, dtype=dtype, keepdims=keepdims)


def prod(a, axis=None, dtype=None, keepdims=False) -> Any:
    """Product of array elements over a given axis."""
    return get_backend().prod(a, axis=axis, dtype=dtype, keepdims=keepdims)


def mean(a, axis=None, dtype=None, keepdims=False) -> Any:
    """Compute the arithmetic mean along the specified axis."""
    return get_backend().mean(a, axis=axis, dtype=dtype, keepdims=keepdims)


def std(a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
    """Compute the standard deviation along the specified axis."""
    return get_backend().std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


def var(a, axis=None, dtype=None, ddof=0, keepdims=False) -> Any:
    """Compute the variance along the specified axis."""
    return get_backend().var(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)


def min(a, axis=None, keepdims=False) -> Any:
    """Return the minimum of an array or minimum along an axis."""
    return get_backend().min(a, axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=False) -> Any:
    """Return the maximum of an array or maximum along an axis."""
    return get_backend().max(a, axis=axis, keepdims=keepdims)


def argmin(a, axis=None) -> Any:
    """Return the indices of the minimum values along an axis."""
    return get_backend().argmin(a, axis=axis)


def argmax(a, axis=None) -> Any:
    """Return the indices of the maximum values along an axis."""
    return get_backend().argmax(a, axis=axis)


def clip(a, a_min, a_max) -> Any:
    """Clip (limit) the values in an array."""
    return get_backend().clip(a, a_min, a_max)


def round(a, decimals=0) -> Any:
    """Round array elements to the given number of decimals."""
    return get_backend().round(a, decimals=decimals)


def exp(x) -> Any:
    """Calculate the exponential of all elements in the input array."""
    return get_backend().exp(x)


def log(x) -> Any:
    """Natural logarithm, element-wise."""
    return get_backend().log(x)


def log10(x) -> Any:
    """Base-10 logarithm, element-wise."""
    return get_backend().log10(x)


def log2(x) -> Any:
    """Base-2 logarithm, element-wise."""
    return get_backend().log2(x)


def log1p(x) -> Any:
    """Natural logarithm of (1 + x), element-wise."""
    return get_backend().log1p(x)


def expm1(x) -> Any:
    """Calculate exp(x) - 1 for all elements in the array."""
    return get_backend().expm1(x)


def sqrt(x) -> Any:
    """Return the non-negative square-root of an array, element-wise."""
    return get_backend().sqrt(x)


def square(x) -> Any:
    """Return the element-wise square of the input."""
    return get_backend().square(x)


def absolute(x) -> Any:
    """Calculate the absolute value element-wise."""
    return get_backend().absolute(x)


def abs(x) -> Any:
    """Calculate the absolute value element-wise."""
    return get_backend().abs(x)


def sign(x) -> Any:
    """Returns an element-wise indication of the sign of a number."""
    return get_backend().sign(x)


def floor(x) -> Any:
    """Return the floor of the input, element-wise."""
    return get_backend().floor(x)


def ceil(x) -> Any:
    """Return the ceiling of the input, element-wise."""
    return get_backend().ceil(x)


def power(x1, x2) -> Any:
    """First array elements raised to powers from second array, element-wise."""
    return get_backend().power(x1, x2)


def isfinite(x) -> Any:
    """Test element-wise for finiteness."""
    return get_backend().isfinite(x)


def isinf(x) -> Any:
    """Test element-wise for positive or negative infinity."""
    return get_backend().isinf(x)


def isnan(x) -> Any:
    """Test element-wise for NaN and return result as a boolean array."""
    return get_backend().isnan(x)


def sin(x) -> Any:
    """Sine, element-wise."""
    return get_backend().sin(x)


def cos(x) -> Any:
    """Cosine, element-wise."""
    return get_backend().cos(x)


def tan(x) -> Any:
    """Tangent, element-wise."""
    return get_backend().tan(x)


def arcsin(x) -> Any:
    """Inverse sine, element-wise."""
    return get_backend().arcsin(x)


def arccos(x) -> Any:
    """Inverse cosine, element-wise."""
    return get_backend().arccos(x)


def arctan(x) -> Any:
    """Inverse tangent, element-wise."""
    return get_backend().arctan(x)


def arctan2(x1, x2) -> Any:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly."""
    return get_backend().arctan2(x1, x2)


def sinh(x) -> Any:
    """Hyperbolic sine, element-wise."""
    return get_backend().sinh(x)


def cosh(x) -> Any:
    """Hyperbolic cosine, element-wise."""
    return get_backend().cosh(x)


def tanh(x) -> Any:
    """Hyperbolic tangent, element-wise."""
    return get_backend().tanh(x)


def arcsinh(x) -> Any:
    """Inverse hyperbolic sine, element-wise."""
    return get_backend().arcsinh(x)


def arccosh(x) -> Any:
    """Inverse hyperbolic cosine, element-wise."""
    return get_backend().arccosh(x)


def arctanh(x) -> Any:
    """Inverse hyperbolic tangent, element-wise."""
    return get_backend().arctanh(x)


def dot(a, b) -> Any:
    """Dot product of two arrays."""
    return get_backend().dot(a, b)


def tensordot(a, b, axes=2) -> Any:
    """Compute tensor dot product along specified axes."""
    return get_backend().tensordot(a, b, axes=axes)


def matmul(a, b) -> Any:
    """Matrix product of two arrays."""
    return get_backend().matmul(a, b)


def kron(a, b) -> Any:
    """Kronecker product of two arrays."""
    return get_backend().kron(a, b)


def outer(a, b) -> Any:
    """Compute the outer product of two vectors."""
    return get_backend().outer(a, b)


def einsum(subscripts, *operands) -> Any:
    """Evaluate the Einstein summation convention on the operands."""
    return get_backend().einsum(subscripts, *operands)


# ============================
# Comparison Functions
# ============================


def equal(x1, x2) -> Any:
    """Return (x1 == x2) element-wise."""
    return get_backend().equal(x1, x2)


def not_equal(x1, x2) -> Any:
    """Return (x1 != x2) element-wise."""
    return get_backend().not_equal(x1, x2)


def greater(x1, x2) -> Any:
    """Return (x1 > x2) element-wise."""
    return get_backend().greater(x1, x2)


def greater_equal(x1, x2) -> Any:
    """Return (x1 >= x2) element-wise."""
    return get_backend().greater_equal(x1, x2)


def less(x1, x2) -> Any:
    """Return (x1 < x2) element-wise."""
    return get_backend().less(x1, x2)


def less_equal(x1, x2) -> Any:
    """Return (x1 <= x2) element-wise."""
    return get_backend().less_equal(x1, x2)


def array_equal(a1, a2) -> Any:
    """True if two arrays have the same shape and elements, False otherwise."""
    return get_backend().array_equal(a1, a2)


def allclose(a, b, rtol=1e-05, atol=1e-08) -> Any:
    """Returns True if two arrays are element-wise equal within a tolerance."""
    return get_backend().allclose(a, b, rtol=rtol, atol=atol)


def all(a, axis=None, keepdims=False) -> Any:
    """Test whether all array elements along a given axis evaluate to True."""
    return get_backend().all(a, axis=axis, keepdims=keepdims)


def any(a, axis=None, keepdims=False) -> Any:
    """Test whether any array element along a given axis evaluates to True."""
    return get_backend().any(a, axis=axis, keepdims=keepdims)


def logical_and(x1, x2) -> Any:
    """Compute the truth value of x1 AND x2 element-wise."""
    return get_backend().logical_and(x1, x2)


def logical_or(x1, x2) -> Any:
    """Compute the truth value of x1 OR x2 element-wise."""
    return get_backend().logical_or(x1, x2)


def logical_not(x) -> Any:
    """Compute the truth value of NOT x element-wise."""
    return get_backend().logical_not(x)


def logical_xor(x1, x2) -> Any:
    """Compute the truth value of x1 XOR x2 element-wise."""
    return get_backend().logical_xor(x1, x2)


# ============================
# Array Type and Information
# ============================


def shape(a) -> Any:
    """Return the shape of an array."""
    return get_backend().shape(a)


def size(a) -> Any:
    """Return the number of elements in an array."""
    return get_backend().size(a)


def ndim(a) -> Any:
    """Return the number of dimensions of an array."""
    return get_backend().ndim(a)


def dtype(a) -> Any:
    """Return the data type of an array."""
    return get_backend().dtype(a)


# ============================
# Linear Algebra Namespace
# ============================


class linalg:
    """Linear algebra functions."""

    @staticmethod
    def inv(a) -> Any:
        """Compute the (multiplicative) inverse of a matrix."""
        return get_backend().inv(a)

    @staticmethod
    def solve(a, b) -> Any:
        """Solve a linear matrix equation, or system of linear scalar equations."""
        return get_backend().solve(a, b)

    @staticmethod
    def eigh(a) -> Any:
        """Return eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix."""
        return get_backend().eigh(a)

    @staticmethod
    def eigvalsh(a) -> Any:
        """Return eigenvalues of a complex Hermitian or real symmetric matrix."""
        return get_backend().eigvalsh(a)

    @staticmethod
    def eigvals(a) -> Any:
        """Return eigenvalues of a general matrix."""
        return get_backend().eigvals(a)

    @staticmethod
    def eig(a) -> Any:
        """Return eigenvalues and eigenvectors of a general matrix."""
        return get_backend().eig(a)

    @staticmethod
    def svd(a, full_matrices=True) -> Any:
        """Singular Value Decomposition."""
        return get_backend().svd(a, full_matrices=full_matrices)

    @staticmethod
    def norm(x, ord=None, axis=None, keepdims=False) -> Any:
        """Matrix or vector norm."""
        return get_backend().norm(x, ord=ord, axis=axis, keepdims=keepdims)

    @staticmethod
    def det(a) -> Any:
        """Compute the determinant of an array."""
        return get_backend().det(a)

    @staticmethod
    def slogdet(a) -> Any:
        """Compute the sign and (natural) logarithm of the determinant of an array."""
        return get_backend().slogdet(a)

    @staticmethod
    def matrix_rank(M, tol=None) -> Any:
        """Return matrix rank of array using SVD method."""
        return get_backend().matrix_rank(M, tol=tol)

    @staticmethod
    def cholesky(a) -> Any:
        """Cholesky decomposition."""
        return get_backend().cholesky(a)

    @staticmethod
    def qr(a, mode="reduced") -> Any:
        """QR decomposition of a matrix."""
        return get_backend().qr(a, mode=mode)

    @staticmethod
    def lstsq(a, b, rcond=None) -> Any:
        """Return the least-squares solution to a linear matrix equation."""
        return get_backend().lstsq(a, b, rcond=rcond)

    @staticmethod
    def pinv(a, rcond=1e-15) -> Any:
        """Compute the (Moore-Penrose) pseudo-inverse of a matrix."""
        return get_backend().pinv(a, rcond=rcond)

    @staticmethod
    def matrix_power(a, n) -> Any:
        """Raise a square matrix to the (integer) power n."""
        return get_backend().matrix_power(a, n)

    @staticmethod
    def multi_dot(arrays) -> Any:
        """Compute the dot product of two or more arrays in a single function call."""
        return get_backend().multi_dot(arrays)


# ============================
# Random Namespace
# ============================


class random:
    """Random number generation functions."""

    @staticmethod
    def seed(seed=None) -> None:
        """Seed the random number generator."""
        return get_backend().random_seed(seed)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw random samples from a normal (Gaussian) distribution."""
        return get_backend().normal(
            loc=loc, scale=scale, size=size, dtype=dtype, key=key
        )

    @staticmethod
    def uniform(low=0.0, high=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw random samples from a uniform distribution."""
        return get_backend().uniform(
            minval=low, maxval=high, size=size, dtype=dtype, key=key
        )

    @staticmethod
    def poisson(lam=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a Poisson distribution."""
        return get_backend().poisson(lam=lam, size=size, dtype=dtype, key=key)

    @staticmethod
    def exponential(scale=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw samples from an exponential distribution."""
        return get_backend().exponential(scale=scale, size=size, dtype=dtype, key=key)

    @staticmethod
    def gamma(shape_param, scale=1.0, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a Gamma distribution."""
        # Pass shape_param as the gamma shape parameter and size as the output shape
        return get_backend().gamma(
            shape=shape_param, scale=scale, size=size, dtype=dtype, key=key
        )

    @staticmethod
    def beta(a, b, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a Beta distribution."""
        return get_backend().beta(a=a, b=b, size=size, dtype=dtype, key=key)

    @staticmethod
    def binomial(n, p, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a binomial distribution."""
        return get_backend().binomial(n=n, p=p, size=size, dtype=dtype, key=key)

    @staticmethod
    def chisquare(df, size=None, dtype=None, key=None) -> Any:
        """Draw samples from a chi-square distribution."""
        return get_backend().chisquare(df=df, size=size, dtype=dtype, key=key)

    @staticmethod
    def randint(low, high=None, size=None, dtype=None, key=None) -> Any:
        """Return random integers from low (inclusive) to high (exclusive)."""
        return get_backend().randint(
            low=low, high=high, size=size, dtype=dtype, key=key
        )

    @staticmethod
    def random(size=None, dtype=None, key=None) -> Any:
        """Return random floats in the half-open interval [0.0, 1.0)."""
        return get_backend().random(size=size, dtype=dtype, key=key)

    @staticmethod
    def randn(*args, dtype=None, key=None) -> Any:
        """Return a sample (or samples) from the "standard normal" distribution."""
        if len(args) == 0:
            return get_backend().normal(size=None, dtype=dtype, key=key)
        else:
            return get_backend().normal(size=args, dtype=dtype, key=key)

    @staticmethod
    def rand(*args, dtype=None, key=None) -> Any:
        """Random values in a given shape."""
        if len(args) == 0:
            return get_backend().uniform(size=None, dtype=dtype, key=key)
        else:
            return get_backend().uniform(size=args, dtype=dtype, key=key)

    @staticmethod
    def choice(a, size=None, replace=True, p=None, key=None) -> Any:
        """Generate a random sample from a given 1-D array."""
        return get_backend().choice(a=a, size=size, replace=replace, p=p, key=key)

    @staticmethod
    def permutation(x, key=None) -> Any:
        """Randomly permute a sequence, or return a permuted range."""
        return get_backend().permutation(x, key=key)

    @staticmethod
    def shuffle(x, key=None) -> Any:
        """Modify a sequence in-place by shuffling its contents."""
        return get_backend().shuffle(x, key=key)

    @staticmethod
    def multivariate_normal(mean, cov, size=None, dtype=None, key=None) -> Any:
        """Draw random samples from a multivariate normal distribution."""
        return get_backend().multivariate_normal(
            mean=mean, cov=cov, size=size, dtype=dtype, key=key
        )


# ============================
# Constants
# ============================

pi = 3.141592653589793
e = 2.718281828459045
inf = float("inf")
nan = float("nan")

# ============================
# Enable direct attribute access for lazy loading
# ============================


class _LazyProxy:
    """Proxy class that forwards attribute access to the active backend."""

    def __getattr__(self, name):
        return getattr(get_backend(), name)


# Create a proxy for lazy attribute access
_proxy = _LazyProxy()


# Enable direct attribute access for things not explicitly defined
def __getattr__(name):
    return getattr(_proxy, name)
