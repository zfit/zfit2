"""Optimization functions for backends."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from . import get_backend


def minimize(
    fun: Callable,
    x0: Any,
    method: Optional[str] = None,
    jac: Optional[Union[bool, Callable]] = None,
    hess: Optional[Union[bool, Callable]] = None,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    constraints: Optional[Sequence[Dict[str, Any]]] = None,
    tol: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Minimize a function using the specified backend.
    
    This function provides a unified interface to various optimization 
    algorithms across different backends.
    
    Args:
        fun: The objective function to minimize.
        x0: Initial guess.
        method: The optimization method to use.
        jac: Method for computing the gradient vector.
        hess: Method for computing the Hessian matrix.
        bounds: Bounds on variables.
        constraints: Constraints definition.
        tol: Tolerance for termination.
        options: Additional options for the optimizer.
        
    Returns:
        A dictionary containing the optimization results.
    """
    backend = get_backend()
    
    if backend.name == "JAX":
        # For JAX, use JAX-compatible optimizers
        from jax import grad, jit
        import jax.numpy as jnp
        import scipy.optimize
        
        # Convert to JAX arrays
        x0_jax = jnp.array(x0)
        
        # Handle gradient computation
        if jac is True:
            gradient = grad(fun)
        elif callable(jac):
            gradient = jac
        else:
            gradient = None
        
        # Define the optimization function
        @jit
        def opt_fun(x):
            return fun(x)
        
        if gradient is not None:
            @jit
            def opt_grad(x):
                return gradient(x)
        else:
            opt_grad = None
        
        # Perform optimization
        result = scipy.optimize.minimize(
            opt_fun,
            x0_jax,
            method=method,
            jac=opt_grad,
            hess=hess,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
        )
        
        # Convert result to dictionary
        return {
            "x": jnp.array(result.x),
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "fun": result.fun,
            "nfev": result.nfev,
            "njev": getattr(result, "njev", 0),
            "nhev": getattr(result, "nhev", 0),
        }
    
    elif backend.name == "NumPy":
        # For NumPy, use SciPy optimizers
        import numpy as np
        from scipy import optimize
        
        # Convert to NumPy arrays
        x0_np = np.array(x0)
        
        # Handle gradient computation
        if jac is True:
            gradient = backend.grad(fun)
        elif callable(jac):
            gradient = jac
        else:
            gradient = None
        
        # Perform optimization
        result = optimize.minimize(
            fun,
            x0_np,
            method=method,
            jac=gradient,
            hess=hess,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            options=options,
        )
        
        # Convert result to dictionary
        return {
            "x": np.array(result.x),
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "fun": result.fun,
            "nfev": result.nfev,
            "njev": getattr(result, "njev", 0),
            "nhev": getattr(result, "nhev", 0),
        }
    
    elif backend.name == "SymPy":
        # SymPy doesn't support numerical optimization in the same way
        from .errors import NotImplementedInBackend
        raise NotImplementedInBackend("minimize", "SymPy")
    
    else:
        raise ValueError(f"Unknown backend: {backend.name}")


def root(
    fun: Callable,
    x0: Any,
    method: Optional[str] = None,
    jac: Optional[Union[bool, Callable]] = None,
    tol: Optional[float] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find the roots of a function using the specified backend.
    
    This function provides a unified interface to various root-finding
    algorithms across different backends.
    
    Args:
        fun: The function for which the roots are sought.
        x0: Initial guess.
        method: The root finding method to use.
        jac: Method for computing the Jacobian matrix.
        tol: Tolerance for termination.
        options: Additional options for the solver.
        
    Returns:
        A dictionary containing the root-finding results.
    """
    backend = get_backend()
    
    if backend.name == "JAX":
        # For JAX, use JAX-compatible optimizers
        from jax import grad, jit, jacfwd
        import jax.numpy as jnp
        import scipy.optimize
        
        # Convert to JAX arrays
        x0_jax = jnp.array(x0)
        
        # Handle Jacobian computation
        if jac is True:
            jacobian = jacfwd(fun)
        elif callable(jac):
            jacobian = jac
        else:
            jacobian = None
        
        # Define the root-finding function
        @jit
        def opt_fun(x):
            return fun(x)
        
        if jacobian is not None:
            @jit
            def opt_jac(x):
                return jacobian(x)
        else:
            opt_jac = None
        
        # Perform root finding
        result = scipy.optimize.root(
            opt_fun,
            x0_jax,
            method=method,
            jac=opt_jac,
            tol=tol,
            options=options,
        )
        
        # Convert result to dictionary
        return {
            "x": jnp.array(result.x),
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "fun": result.fun,
            "nfev": result.nfev,
            "njev": getattr(result, "njev", 0),
        }
    
    elif backend.name == "NumPy":
        # For NumPy, use SciPy optimizers
        import numpy as np
        from scipy import optimize
        
        # Convert to NumPy arrays
        x0_np = np.array(x0)
        
        # Handle Jacobian computation
        if jac is True:
            jacobian = backend.jacobian(fun)
        elif callable(jac):
            jacobian = jac
        else:
            jacobian = None
        
        # Perform root finding
        result = optimize.root(
            fun,
            x0_np,
            method=method,
            jac=jacobian,
            tol=tol,
            options=options,
        )
        
        # Convert result to dictionary
        return {
            "x": np.array(result.x),
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "fun": result.fun,
            "nfev": result.nfev,
            "njev": getattr(result, "njev", 0),
        }
    
    elif backend.name == "SymPy":
        # SymPy has symbolic solvers, but they work differently
        import sympy as sp
        
        # For now, we'll only support scalar equations with a single variable
        if not hasattr(x0, "__len__") or len(x0) == 1:
            # Create a symbolic variable
            x = sp.symbols('x')
            
            # Convert the function to a symbolic expression
            try:
                # Try to evaluate the function symbolically
                expr = fun(x)
                
                # Solve the equation
                solutions = sp.solve(expr, x)
                
                return {
                    "x": solutions,
                    "success": len(solutions) > 0,
                    "message": f"Found {len(solutions)} solutions",
                }
            except Exception as e:
                return {
                    "x": None,
                    "success": False,
                    "message": str(e),
                }
        else:
            from .errors import NotImplementedInBackend
            raise NotImplementedInBackend("root for multivariable functions", "SymPy")
    
    else:
        raise ValueError(f"Unknown backend: {backend.name}")


def curve_fit(
    f: Callable,
    xdata: Any,
    ydata: Any,
    p0: Optional[Any] = None,
    sigma: Optional[Any] = None,
    absolute_sigma: bool = False,
    method: Optional[str] = None,
    jac: Optional[Union[bool, Callable]] = None,
    bounds: Optional[Tuple[Any, Any]] = None,
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    gtol: Optional[float] = None,
    max_nfev: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Use non-linear least squares to fit a function to data.
    
    This function provides a unified interface for curve fitting
    across different backends.
    
    Args:
        f: The model function, f(x, ...). It must take the independent
           variable as the first argument and the parameters to fit as
           separate remaining arguments.
        xdata: The independent variable where the data is measured.
        ydata: The dependent data.
        p0: Initial guess for the parameters.
        sigma: Uncertainties in ydata.
        absolute_sigma: If True, sigma is used in an absolute sense and the
                        estimated parameter covariance reflects these absolute
                        values.
        method: Method to use for optimization.
        jac: Function with signature jac(x, ...) to compute the Jacobian matrix.
        bounds: Lower and upper bounds on parameters.
        ftol, xtol, gtol: Tolerances for termination.
        max_nfev: Maximum number of function evaluations.
        
    Returns:
        Tuple with optimal parameters and estimated covariance.
    """
    backend = get_backend()
    
    if backend.name == "JAX":
        # For JAX, use JAX-compatible curve_fit
        from jax import grad, jit, jacfwd
        import jax.numpy as jnp
        import scipy.optimize
        
        # Convert to JAX arrays
        xdata_jax = jnp.array(xdata)
        ydata_jax = jnp.array(ydata)
        
        if p0 is not None:
            p0_jax = jnp.array(p0)
        else:
            p0_jax = None
        
        if sigma is not None:
            sigma_jax = jnp.array(sigma)
        else:
            sigma_jax = None
        
        # Handle Jacobian computation
        if jac is True:
            # Define a wrapper to compute the Jacobian with respect to parameters
            def model_wrapper(params):
                def wrapped(x):
                    return f(x, *params)
                return wrapped
            
            def jac_wrapper(params, x):
                return jacfwd(model_wrapper(params))(x)
        elif callable(jac):
            jac_wrapper = jac
        else:
            jac_wrapper = None
        
        # Perform curve fitting
        popt, pcov = scipy.optimize.curve_fit(
            f,
            xdata_jax,
            ydata_jax,
            p0=p0_jax,
            sigma=sigma_jax,
            absolute_sigma=absolute_sigma,
            method=method,
            jac=jac_wrapper,
            bounds=bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        
        return jnp.array(popt), jnp.array(pcov)
    
    elif backend.name == "NumPy":
        # For NumPy, use SciPy optimizers
        import numpy as np
        from scipy import optimize
        
        # Convert to NumPy arrays
        xdata_np = np.array(xdata)
        ydata_np = np.array(ydata)
        
        if p0 is not None:
            p0_np = np.array(p0)
        else:
            p0_np = None
        
        if sigma is not None:
            sigma_np = np.array(sigma)
        else:
            sigma_np = None
        
        # Handle Jacobian computation
        if jac is True:
            # Let scipy handle automatic differentiation
            jac_wrapper = '2-point'
        elif callable(jac):
            jac_wrapper = jac
        else:
            jac_wrapper = None
        
        # Perform curve fitting
        popt, pcov = optimize.curve_fit(
            f,
            xdata_np,
            ydata_np,
            p0=p0_np,
            sigma=sigma_np,
            absolute_sigma=absolute_sigma,
            method=method,
            jac=jac_wrapper,
            bounds=bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_nfev,
        )
        
        return np.array(popt), np.array(pcov)
    
    elif backend.name == "SymPy":
        # SymPy doesn't support numerical curve fitting in the same way
        from .errors import NotImplementedInBackend
        raise NotImplementedInBackend("curve_fit", "SymPy")
    
    else:
        raise ValueError(f"Unknown backend: {backend.name}")
