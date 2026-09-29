"""Shared exception types."""
from __future__ import annotations


class JobCancelled(Exception):
    """Raised inside the worker when a job is cancelled (e.g. forced delete)."""


class ProcessingError(Exception):
    """Raised when a video cannot be processed."""


class JobError(Exception):
    """An API-level error that is rendered as a consistent JSON error response."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
