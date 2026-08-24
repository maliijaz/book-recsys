"""Backend settings, sourced from environment variables so the same image
works unmodified in local Docker Compose and on the deployed host (Render)."""
import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/app/artifacts"))

# The deployed Vercel frontend origin(s) must be listed here (comma-separated)
# via an env var, or cross-origin calls from the frontend will be blocked by
# the browser -- no host enables this for you by default.
_default_origins = "http://localhost:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
