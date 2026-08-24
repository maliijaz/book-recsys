"""Backend settings, sourced from environment variables so the same code
works unmodified locally, in Docker, and as a Vercel Python function."""
import os
from pathlib import Path

# Default resolves relative to this file (backend/app/config.py -> backend/artifacts),
# which works with zero config wherever the process's cwd ends up (Vercel
# Functions, `uvicorn` run from any directory). ARTIFACTS_DIR still overrides
# it for Docker, where the image copies artifacts to a fixed /app/artifacts.
_DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(_DEFAULT_ARTIFACTS_DIR)))

# The deployed Vercel frontend origin(s) must be listed here (comma-separated)
# via an env var, or cross-origin calls from the frontend will be blocked by
# the browser -- no host enables this for you by default.
_default_origins = "http://localhost:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
