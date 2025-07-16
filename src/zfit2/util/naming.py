from __future__ import annotations

import keyword
import re
from typing import Any


def is_valid_name(obj: Any, raise_error: bool = True) -> bool | None:
    """
    Check if an object is a valid name string.

    A valid name must:
    - Be a string
    - Be a valid Python identifier with additional allowed characters (. and :)
    - Not start with . or :

    Args:
        obj: Object to validate
        raise_error: If True, raise exceptions on invalid names. If False, return boolean.

    Returns:
        True if valid (when raise_error=False), None if valid (when raise_error=True)
        False if invalid (when raise_error=False)

    Raises:
        TypeError: If obj is not a string (only when raise_error=True)
        ValueError: If the name is invalid with specific reason (only when raise_error=True)
    """
    if not isinstance(obj, str):
        if raise_error:
            msg = f"Name must be a string, got {type(obj).__name__}"
            raise TypeError(msg)
        return False

    if not obj:
        if raise_error:
            msg = "Name cannot be empty"
            raise ValueError(msg)
        return False

    if obj.startswith((".", ":")):
        if raise_error:
            msg = "Name cannot start with '.' or ':'"
            raise ValueError(msg)
        return False

    # Check if it's a valid Python identifier allowing . and : in the middle
    # Replace . and : temporarily to check the rest
    cleaned_name = re.sub(r"[.:]", "_", obj)

    if not cleaned_name.isidentifier():
        if raise_error:
            msg = f"'{obj}' is not a valid identifier"
            raise ValueError(msg)
        return False

    if keyword.iskeyword(cleaned_name):
        if raise_error:
            msg = f"'{obj}' is a Python keyword and cannot be used as a name"
            raise ValueError(msg)
        return False

    # Additional check: ensure . and : are not at invalid positions
    if obj.endswith((".", ":")):
        if raise_error:
            msg = "Name cannot end with '.' or ':'"
            raise ValueError(msg)
        return False

    # Check for consecutive special characters
    if ".." in obj or "::" in obj or ".:" in obj or ":." in obj:
        if raise_error:
            msg = "Name cannot contain consecutive special characters"
            raise ValueError(msg)
        return False

    return True
