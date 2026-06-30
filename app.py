"""
Vercel Python runtime entry point :).

Vercel expects a module-level `app` object when using the Python runtime.
This file re-exports the FastAPI app defined in `apps/api/main.py` so
Vercel can discover it without needing to know the internal package layout.
"""

from apps.api.main import app

__all__ = ["app"]
