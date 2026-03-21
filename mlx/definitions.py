from typing import Any, Callable, Dict, List, Optional, Tuple

ModuleConfig = Dict[str, Any]


class MLXUserError(Exception):
    """Raised for user-facing configuration or runtime validation errors."""


class MLXAbort(Exception):
    """Raised when an interactive action is cancelled by the user."""
