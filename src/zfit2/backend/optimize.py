"""Optimization utilities for the backend module."""

from __future__ import annotations

from collections.abc import Callable


def minimize(fun: Callable, x0, method=None, jac=None, hess=None, **kwargs):
    """Minimize a function using the current backend.

    This function provides a unified interface to optimization routines
    across different backends. The actual implementation used depends
    on the active backend.

    Args:
        fun: The objective function to minimize.
        x0: Initial guess.
        method: The optimization method to use.
        jac: The Jacobian of the objective function.
        hess: The Hessian of the objective function.
        **kwargs: Additional arguments to pass to the backend-specific optimizer.

    Returns:
        The optimization result object.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.optimize import minimize
        >>>
        >>> def f(x):
        ...     return x[0]**2 + x[1]**2
        >>>
        >>> # Minimize using the current backend
        >>> result = minimize(f, [1.0, 1.0])
        >>> result.x  # Optimal parameters
    """
    from . import get_backend

    backend = get_backend()

    # Handle default gradient calculation
    if jac is None and hasattr(backend, "grad"):
        jac = backend.grad(fun)

    # Handle default Hessian calculation
    if (
        hess is None
        and method
        and "newton" in method.lower()
        and hasattr(backend, "hessian")
    ):
        hess = backend.hessian(fun)

    # Use backend-specific implementation
    if backend.name == "JAX":
        from jax.scipy.optimize import minimize as jax_minimize

        return jax_minimize(fun, x0, method=method, jac=jac, hess=hess, **kwargs)

    elif backend.name == "NumPy":
        from scipy.optimize import minimize as scipy_minimize

        return scipy_minimize(fun, x0, method=method, jac=jac, hess=hess, **kwargs)

    elif backend.name == "SymPy":
        # For SymPy, we'll provide a simplified implementation that converts
        # to symbolic expressions, computes critical points, and returns the minimum
        import numpy as np
        import sympy as sp

        # Convert to symbolic
        if np.isscalar(x0) or len(x0) == 1:
            # Scalar case
            x_sym = sp.Symbol("x")

            # Define symbolic function
            def symbolic_fun(x_val):
                return fun(np.array([x_val]))

            # Find critical points by solving for f'(x) = 0
            expr = symbolic_fun(x_sym)
            deriv = sp.diff(expr, x_sym)
            critical_points = sp.solve(deriv, x_sym)

            # Evaluate function at all critical points
            if critical_points:
                values = [symbolic_fun(cp) for cp in critical_points if sp.im(cp) == 0]
                if values:
                    min_idx = np.argmin(values)
                    min_point = critical_points[min_idx]
                    min_value = values[min_idx]

                    # Create a result object similar to scipy's
                    class OptimizeResult:
                        pass

                    result = OptimizeResult()
                    result.x = np.array([float(min_point)])
                    result.fun = float(min_value)
                    result.success = True
                    return result

            # Fallback: return the initial point
            class OptimizeResult:
                pass

            result = OptimizeResult()
            result.x = x0
            result.fun = fun(x0)
            result.success = False
            result.message = "Symbolic minimization failed"
            return result

        else:
            # Multivariate case - not fully implemented in SymPy backend
            raise NotImplementedError(
                "Multivariate optimization not implemented for SymPy backend. "
                "Please use 'numpy' or 'jax' backend for optimization."
            )

    else:
        raise ValueError(f"Unsupported backend: {backend.name}")


def curve_fit(
    f: Callable,
    xdata,
    ydata,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    method=None,
    jac=None,
    **kwargs,
):
    """Use non-linear least squares to fit a function to data.

    This function provides a unified interface to curve fitting routines
    across different backends. The actual implementation used depends
    on the active backend.

    Args:
        f: The model function, f(x, ...). It must take the independent variable
           as the first argument and the parameters to fit as separate remaining arguments.
        xdata: The independent variable where the data is measured.
        ydata: The dependent data.
        p0: Initial guess for the parameters.
        sigma: Determines the uncertainty in ydata.
        absolute_sigma: If True, sigma is used in an absolute sense.
        method: The optimization method to use.
        jac: Function with signature jac(x, ...) to compute the Jacobian matrix.
        **kwargs: Additional arguments to pass to the backend-specific optimizer.

    Returns:
        Optimal values for the parameters and estimated covariance.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.optimize import curve_fit
        >>>
        >>> def f(x, a, b):
        ...     return a * x + b
        >>>
        >>> xdata = znp.array([0, 1, 2, 3, 4])
        >>> ydata = znp.array([1, 3, 5, 7, 9])
        >>>
        >>> # Fit a line to the data points
        >>> popt, pcov = curve_fit(f, xdata, ydata)
        >>> popt  # Should be close to [2, 1]
    """
    from . import get_backend

    backend = get_backend()

    # Use backend-specific implementation
    if backend.name == "JAX":
        # JAX doesn't have a dedicated curve_fit, so we'll implement it
        # using JAX's optimizer and autodiff capabilities
        import jax
        import jax.numpy as jnp

        # Create the least squares objective function
        def objective(params):
            # Apply the model function to the data
            model = f(xdata, *params)

            # Handle sigma if provided
            if sigma is not None:
                if absolute_sigma:
                    # Absolute sigma values
                    residuals = (ydata - model) / sigma
                else:
                    # Relative sigma values
                    scale = jnp.sum(((ydata - model) / sigma) ** 2) / (
                        len(ydata) - len(params)
                    )
                    residuals = (ydata - model) / (sigma * jnp.sqrt(scale))
            else:
                residuals = ydata - model

            # Sum of squared residuals
            return jnp.sum(residuals**2)

        # Handle initial guess
        if p0 is None:
            # Create a default initial guess of ones
            # We need to determine how many parameters the function expects
            from inspect import signature

            sig = signature(f)
            n_params = len(sig.parameters) - 1  # Subtract 1 for xdata
            p0 = jnp.ones(n_params)

        # Optimize
        from jax.scipy.optimize import minimize

        result = minimize(objective, p0, method=method or "BFGS")

        # Estimate covariance matrix
        # This is a simplified approach; a more accurate version would
        # compute the Hessian at the optimum
        hess_fun = jax.hessian(objective)
        hessian_matrix = hess_fun(result.x)
        try:
            # Invert the Hessian to get the covariance matrix
            pcov = jnp.linalg.inv(hessian_matrix)
        except:
            # If inversion fails, return a matrix of inf
            pcov = jnp.full((len(result.x), len(result.x)), jnp.inf)

        return result.x, pcov

    elif backend.name == "NumPy":
        from scipy.optimize import curve_fit as scipy_curve_fit

        return scipy_curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            method=method,
            jac=jac,
            **kwargs,
        )

    elif backend.name == "SymPy":
        # For SymPy, we'll provide a simplified implementation
        # that can work with symbolic expressions
        import numpy as np
        import sympy as sp

        # Convert to numpy arrays
        xdata_np = np.asarray(xdata)
        ydata_np = np.asarray(ydata)

        # Create symbolic parameters
        from inspect import signature

        sig = signature(f)
        n_params = len(sig.parameters) - 1  # Subtract 1 for xdata

        param_names = list(sig.parameters.keys())[1:]  # Skip first parameter (xdata)
        params = sp.symbols(param_names)

        # Create symbolic x
        x_sym = sp.Symbol("x")

        # Create symbolic model function
        expr = f(x_sym, *params)

        # Create the least squares objective function
        residuals = [
            (expr.subs(x_sym, x_val) - y_val) ** 2
            for x_val, y_val in zip(xdata_np, ydata_np, strict=False)
        ]
        obj_expr = sum(residuals)

        # Take the derivatives with respect to all parameters
        derivs = [sp.diff(obj_expr, param) for param in params]

        # Create a system of equations by setting all derivatives to zero
        system = derivs

        # Try to solve the system
        try:
            solution = sp.solve(system, params, dict=True)

            if solution:
                # Get the first solution
                sol = solution[0]

                # Convert to numpy array
                popt = np.array([float(sol[param]) for param in params])

                # Compute the Hessian for covariance estimation
                hessian = [
                    [sp.diff(obj_expr, p1, p2) for p1 in params] for p2 in params
                ]

                # Substitute the solution into the Hessian
                hessian_at_sol = [[float(h.subs(sol)) for h in row] for row in hessian]
                hessian_np = np.array(hessian_at_sol)

                # Invert the Hessian to get the covariance matrix
                try:
                    pcov = np.linalg.inv(hessian_np)
                except:
                    pcov = np.full((n_params, n_params), np.inf)

                return popt, pcov
        except:
            pass

        # If symbolic solving fails, fall back to numerical optimization
        print("Symbolic curve fitting failed, falling back to numerical optimization")
        from scipy.optimize import curve_fit as scipy_curve_fit

        return scipy_curve_fit(
            lambda x, *params: float(f(x, *params)),
            xdata_np,
            ydata_np,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            method=method,
            jac=jac,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported backend: {backend.name}")


def root(fun: Callable, x0, method=None, jac=None, **kwargs):
    """Find the roots of a function using the current backend.

    This function provides a unified interface to root-finding routines
    across different backends. The actual implementation used depends
    on the active backend.

    Args:
        fun: The function for which to find roots.
        x0: Initial guess.
        method: The root-finding method to use.
        jac: The Jacobian of the function.
        **kwargs: Additional arguments to pass to the backend-specific root finder.

    Returns:
        The root-finding result object.

    Example:
        >>> import zfit2.backend as z
        >>> from zfit2.backend import numpy as znp
        >>> from zfit2.backend.optimize import root
        >>>
        >>> def f(x):
        ...     return [x[0]**2 - x[1] - 1, x[1]**2 - x[0] - 1]
        >>>
        >>> # Find roots using the current backend
        >>> result = root(f, [1.0, 1.0])
        >>> result.x  # Root of the function
    """
    from . import get_backend

    backend = get_backend()

    # Handle default Jacobian calculation
    if jac is None and hasattr(backend, "jacobian"):
        jac = backend.jacobian(fun)

    # Use backend-specific implementation
    if backend.name == "JAX":
        # JAX doesn't have a dedicated root finder, so we'll use optimization
        def objective(x):
            return backend.sum(backend.square(fun(x)))

        from jax.scipy.optimize import minimize as jax_minimize

        result = jax_minimize(objective, x0, method="BFGS", jac=None)

        # Create a result object similar to scipy's root
        if hasattr(result, "x"):
            result.fun = fun(result.x)
            result.success = result.success and backend.all(
                backend.abs(result.fun) < 1e-6
            )

        return result

    elif backend.name == "NumPy":
        from scipy.optimize import root as scipy_root

        return scipy_root(fun, x0, method=method, jac=jac, **kwargs)

    elif backend.name == "SymPy":
        # For SymPy, we'll provide a simplified implementation that converts
        # to symbolic expressions and solves the system of equations
        import numpy as np
        import sympy as sp

        # Convert to symbolic
        if np.isscalar(x0) or len(x0) == 1:
            # Scalar case
            x_sym = sp.Symbol("x")

            # Define symbolic function
            def symbolic_fun(x_val):
                return fun(np.array([x_val]))

            # Solve f(x) = 0
            expr = symbolic_fun(x_sym)
            roots = sp.solve(expr, x_sym)

            # Find the root closest to the initial guess
            if roots:
                real_roots = [root for root in roots if sp.im(root) == 0]
                if real_roots:
                    diffs = [abs(float(root) - x0[0]) for root in real_roots]
                    closest_idx = np.argmin(diffs)
                    closest_root = real_roots[closest_idx]

                    # Create a result object similar to scipy's
                    class OptimizeResult:
                        pass

                    result = OptimizeResult()
                    result.x = np.array([float(closest_root)])
                    result.fun = np.array([0.0])
                    result.success = True
                    return result

            # Fallback: return the initial point
            class OptimizeResult:
                pass

            result = OptimizeResult()
            result.x = x0
            result.fun = fun(x0)
            result.success = False
            result.message = "Symbolic root finding failed"
            return result

        else:
            # Multivariate case - not fully implemented in SymPy backend
            raise NotImplementedError(
                "Multivariate root finding not implemented for SymPy backend. "
                "Please use 'numpy' or 'jax' backend for root finding."
            )

    else:
        raise ValueError(f"Unsupported backend: {backend.name}")
