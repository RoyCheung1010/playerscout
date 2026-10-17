"""Domain exceptions shared across the pipeline, models and UI."""

from __future__ import annotations


class InsufficientDataError(ValueError):
    """Raised when a cohort is too small for the requested operation.

    The dashboard catches this to show a "relax your filters" message instead
    of a stack trace.
    """


class TargetPlayerNotFoundError(LookupError):
    """Raised when a similarity search target is not in the cohort."""
