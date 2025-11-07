"""
QuanTa package initialization.

Exposes the current package version and convenience imports where relevant.
"""

from importlib import metadata


def __getattr__(name):
    if name == "__version__":
        try:
            return metadata.version("QuanTa")
        except metadata.PackageNotFoundError:
            return "0.0.0"
    raise AttributeError(f"module 'quanta' has no attribute {name!r}")


__all__ = ["__version__"]
