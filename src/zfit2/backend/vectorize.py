"""Vectorization utilities for backend operations."""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from . import get_backend


def vmap(
    fun: Callable, 
    in_axes: Union[int, Sequence[int], None] = 0, 
    out_axes: Union[int, None] = 0
) -> Callable:
    """Vectorizes a function along the specified axes.
    
    This function is an abstraction over different backend implementations:
    - For JAX, it uses jax.vmap directly
    - For NumPy, it uses numpy.vectorize or a custom implementation
    - For SymPy, it raises NotImplementedInBackend
    
    Args:
        fun: Function to be vectorized.
        in_axes: Specifies which axes of inputs to map over, either:
            - Integer: Specifies the same axis for all inputs
            - Sequence: Specifies an axis for each input
            - None: Specifies that the input is broadcasted
        out_axes: Specifies which axis the output should have.
            
    Returns:
        A vectorized version of the input function.
    
    Example:
        >>> import zfit2.backend as zb
        >>> def f(x):
        ...     return x ** 2
        >>> vf = zb.vmap(f)
        >>> x = zb.array([1, 2, 3])
        >>> vf(x)  # array([1, 4, 9])
    """
    backend = get_backend()
    
    if backend.name == "JAX":
        # Use JAX's vmap directly
        return backend._vmap(fun, in_axes=in_axes, out_axes=out_axes)
    
    elif backend.name == "NumPy":
        # Use a custom implementation for NumPy
        import numpy as np
        
        # Handle in_axes
        if in_axes is None:
            # Broadcast the input
            def vectorized_fun(*args, **kwargs):
                return fun(*args, **kwargs)
        elif isinstance(in_axes, int):
            # Same axis for all inputs
            def vectorized_fun(*args, **kwargs):
                # Get the shape of the input along the specified axis
                if not args:
                    return fun(*args, **kwargs)
                
                shape = np.array(args[0]).shape[in_axes]
                result = np.zeros(shape)
                
                # Apply function to each element
                for i in range(shape):
                    new_args = []
                    for arg in args:
                        arg_array = np.array(arg)
                        # Handle scalar arguments
                        if arg_array.ndim <= in_axes:
                            new_args.append(arg)
                        else:
                            # Select the current slice
                            idx = [slice(None)] * arg_array.ndim
                            idx[in_axes] = i
                            new_args.append(arg_array[tuple(idx)])
                    
                    result[i] = fun(*new_args, **kwargs)
                
                return result
        else:
            # Different axis for each input
            def vectorized_fun(*args, **kwargs):
                if not args or len(args) != len(in_axes):
                    return fun(*args, **kwargs)
                
                # Get the shape of the output
                shapes = []
                for arg, axis in zip(args, in_axes):
                    if axis is not None:
                        arg_array = np.array(arg)
                        if arg_array.ndim > axis:
                            shapes.append(arg_array.shape[axis])
                
                if not shapes:
                    return fun(*args, **kwargs)
                
                shape = shapes[0]
                result = np.zeros(shape)
                
                # Apply function to each element
                for i in range(shape):
                    new_args = []
                    for arg, axis in zip(args, in_axes):
                        arg_array = np.array(arg)
                        if axis is None or arg_array.ndim <= axis:
                            new_args.append(arg)
                        else:
                            # Select the current slice
                            idx = [slice(None)] * arg_array.ndim
                            idx[axis] = i
                            new_args.append(arg_array[tuple(idx)])
                    
                    result[i] = fun(*new_args, **kwargs)
                
                return result
        
        return vectorized_fun
    
    elif backend.name == "SymPy":
        # SymPy doesn't support vectorization
        from .errors import NotImplementedInBackend
        raise NotImplementedInBackend("vmap", "SymPy")
    
    else:
        raise ValueError(f"Unknown backend: {backend.name}")
