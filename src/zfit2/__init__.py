"""Copyright (c) 2024 Jonas Eschle. All rights reserved.

zfit2: Scalable pythonic likelihood fitting for science
"""

from __future__ import annotations

import os

# Check JAX_ENABLE_X64 environment variable
jax_enable_x64 = os.environ.get("JAX_ENABLE_X64")
if jax_enable_x64 == "false":
    msg = "zfit2 requires JAX_ENABLE_X64 to be enabled. Please set JAX_ENABLE_X64=true or remove the environment variable."
    raise RuntimeError(msg)
elif jax_enable_x64 is None:
    os.environ["JAX_ENABLE_X64"] = "true"

from ._version import version as __version__

__all__ = ["__version__"]
