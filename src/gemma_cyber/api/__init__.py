"""HTTP API + web service over the shared inference engine (Phase 7/8).

The service is a thin transport over :mod:`gemma_cyber.inference`; it holds no
model logic. Import :func:`create_app` to build the FastAPI application.

Requires the ``api`` extra (``pip install -e '.[api]'``). FastAPI/uvicorn are
imported lazily inside :func:`create_app` so the core package stays web-free.
"""

from __future__ import annotations

from gemma_cyber.api.app import create_app

__all__ = ["create_app"]
