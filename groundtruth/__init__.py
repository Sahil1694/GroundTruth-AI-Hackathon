"""
GroundTruth H-002 Intelligent Customer Experience Agent package.

This package contains the modular RAG backend implementation that powers
the recommendation service exposed through the FastAPI application.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("groundtruth")
except PackageNotFoundError:  # pragma: no cover - package metadata missing in dev
    __version__ = "0.0.0"

__all__ = ["__version__"]

