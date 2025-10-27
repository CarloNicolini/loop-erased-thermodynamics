"""Compatibility shim: re-export the unified ``Wilson`` estimator.

Older code imported ``Wilson`` from ``wilson2``. We keep that import
working by re-exporting the new unified ``Wilson`` class from
``wilson.wilson``.
"""

from .wilson import Wilson as Wilson
