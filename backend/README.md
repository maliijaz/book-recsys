# Book Recommender API

FastAPI backend for the book recommender product. Serves precomputed
persona recommendations and a live re-ranking endpoint from artifacts built
by the `pipeline` package -- no database. See `/docs` for the interactive
API reference once deployed.

## Deploying (Render, free)

1. Push this repo to GitHub with `artifacts/` populated (run
   `python -m pipeline.cli run-all` first, or `docker compose run --rm pipeline`).
2. On [Render](https://render.com), create a new **Web Service** directly
   (not via **Blueprint** -- the Blueprint flow can prompt for card
   verification even for a free-plan service; a plain Web Service does
   not). Connect the repo, set **Root Directory** to the repo root, and
   **Dockerfile Path** to `backend/Dockerfile` (Docker build context stays
   the repo root so it can `COPY artifacts ./artifacts`). Select the
   **Free** instance type.
3. Set the environment variable `CORS_ALLOWED_ORIGINS` to your deployed
   frontend origin (e.g. `https://your-app.vercel.app`).
4. Render injects its own `$PORT` at runtime and the container's
   `CMD` already binds to it -- no extra config needed.

`render.yaml` in the repo root documents the same config as Infrastructure-
as-Code for reference (`render blueprint launch` from the Render CLI, or
Render's Blueprint UI), but isn't required -- the manual Web Service path
above is the one confirmed not to require a card.

Free tier is 512MB RAM / 0.1 CPU and sleeps after 15 minutes idle (~30-50s
cold start on the next request); this backend's own memory footprint is
~150MB with all artifacts loaded, well within that limit.
