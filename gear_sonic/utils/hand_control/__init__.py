"""Shared Inspire adapter API, optionally reusable by other hand integrations.

See README.md for integration guidance. This package does not register hands;
the existing Dex3 deployment uses its original implementation.
"""

from .interface import HandBackend, HandDescription, HandState

__all__ = ["HandBackend", "HandDescription", "HandState"]
