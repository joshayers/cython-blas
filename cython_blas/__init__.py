"""Cython BLAS."""

import contextlib

with contextlib.suppress(ImportError):
    from cython_blas import _init_local  # noqa: F401
