"""Errors raised by the local BiyoVes application."""


class RuntimeErrorBase(RuntimeError):
    """Base class for user-facing runtime failures."""


class ModelAssetsError(RuntimeErrorBase):
    """Raised when required local model assets are unavailable or invalid."""


class InputProcessingError(RuntimeErrorBase):
    """Raised for deterministic per-image processing failures."""
