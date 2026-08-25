# Book Recommender API

FastAPI backend for the book recommender product. Serves precomputed
persona recommendations and a live re-ranking endpoint from artifacts built
by the `pipeline` package -- no database. See `/docs` for the interactive
API reference once deployed.

Live at: [shelf-recs-api.vercel.app/docs](https://shelf-recs-api.vercel.app/docs)

## Deploying (Vercel, free)

This directory is self-contained (`artifacts/` lives here) so it deploys as
its own Vercel project with zero extra config:

```bash
cd backend
vercel deploy --yes --prod
```

Vercel auto-detects the FastAPI `app` instance at `app/main.py`. After the
first deploy, set the `CORS_ALLOWED_ORIGINS` environment variable to your
frontend's URL and redeploy:

```bash
vercel env add CORS_ALLOWED_ORIGINS production --value "https://your-frontend.vercel.app" --yes
vercel deploy --yes --prod --force
```

`vercel.json` sets `maxDuration: 30` for the function. Vercel Hobby is free
with no credit card required.

## Alternative: Docker / self-hosting

`Dockerfile` builds a standalone container (`uvicorn`, binds to `$PORT`) for
any Docker host -- used locally by `docker-compose.yml` at the repo root.
We tried Hugging Face Spaces and Render as free hosts for this before
settling on Vercel: HF changed policy in mid-2026 so Docker/Gradio SDK
Spaces now require a paid PRO plan, and Render prompted for card
verification in practice on both its Blueprint and plain Web Service
creation flows, despite that not being a documented requirement. If you
have access to a different free Docker host, this image should work
unmodified -- just set `ARTIFACTS_DIR` (defaults to `./artifacts` relative
to this directory) and `CORS_ALLOWED_ORIGINS`.
