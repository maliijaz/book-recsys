# Shelf — a deep-learning book recommendation product

A full-stack book recommender built on **goodbooks-10k**: an offline ML pipeline
(popularity → collaborative filtering → content embeddings → Neural
Collaborative Filtering → a two-tower retrieval model → a cold-start-aware
hybrid), served by a database-free FastAPI backend, in front of a Next.js
web app. No login system — instead, a visitor can pick a few books they like
and get freshly re-ranked recommendations computed live from precomputed
embeddings, no account required.

Built entirely on free, open-source datasets, libraries, and hosting.

- **Live demo:** _add your deployed Vercel URL here after deploying_
- **API:** _add your deployed Render URL here after deploying_

## Architecture

```
Next.js 16 (Vercel, free)  ──HTTPS──▶  FastAPI (Render, Docker, free)
                                           │ loads at startup, no DB
                                           ▼
                              artifacts/ baked into the image:
                              book_metadata.parquet, item_embeddings.npy,
                              content_embeddings.npy, cf_weight.npy,
                              persona_recommendations.json, metrics.json
                                           ▲
                              produced offline by `python -m pipeline.cli run-all`
```

- **`pipeline/`** — a real, testable Python package (not notebooks) that downloads
  goodbooks-10k, trains all 8 models, evaluates them, and exports the
  artifacts above.
- **`backend/`** — FastAPI service. Deliberately excludes torch/sentence-transformers/etc:
  it only does numpy/pandas lookups against artifacts the pipeline already
  produced. Two serving paths:
  - *Batch*: precomputed recommendations for ~20 sample "persona" users from the dataset.
  - *Live*: `POST /recommendations/live` — pick a few books, no login, and the backend
    mean-pools their embeddings into an ad-hoc user vector and re-ranks the whole
    catalog on the fly.
- **`frontend/`** — Next.js 16 App Router + Tailwind v4. Catalog browse/search, book
  detail with a similar-books rail, persona browsing, the live taste-profile flow,
  and a model-card page rendering the offline metrics.

## Dataset

[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) — 10,000 books,
53,424 users, ~6M ratings (1–5), plus metadata, genre/shelf tags. Free, no
access gating.

**Known limitation:** `ratings.csv` has no timestamps, so the evaluation
split is leave-one-out per user (not time-based), and no genuine sequential
model is trained on this dataset. See `pipeline/data/bookcrossing.py` for an
optional secondary dataset (Book-Crossing, matched by ISBN) used only to
build a more severe, independent cold-start evaluation slice.

## Model lineup

| # | Model | Approach |
|---|---|---|
| 1 | Popularity | Bayesian-averaged rating count/quality baseline |
| 2 | Item k-NN CF | Cosine similarity over the item-user rating matrix |
| 3 | MF (explicit) | `scikit-surprise` SVD |
| 4 | MF (implicit) | `implicit` ALS on confidence-weighted ratings |
| 5 | Content embeddings | `sentence-transformers` (all-MiniLM-L6-v2) over title/author/tags |
| 6 | NCF | PyTorch, fused GMF + MLP (He et al.) |
| 7 | **Two-Tower** | PyTorch, in-batch sampled softmax; item tower fuses an id embedding with a projected content embedding, so even zero-interaction books get a usable vector |
| 8 | **Hybrid (deployed)** | Per-item blend of two-tower + content score; falls back to pure content similarity for books with < 5 training ratings |

Full evaluation methodology (Precision/Recall/NDCG/HitRate@{5,10,20}, catalog
coverage, and a dedicated cold-start slice) lives in `pipeline/evaluation/`.
Results are exported to `artifacts/metrics.json` and rendered at `/about` in
the frontend.

### Actual results (trained on the real dataset)

goodbooks-10k turns out to be densely rated by construction — every item has
≥8 and every user has ≥18 training ratings, so a naive "<5 interactions"
cold-start definition was empty. Cold items/users here are instead defined
as the bottom ~5% of items (<100 train ratings) and bottom ~1% of users
(<40 train ratings) by interaction count (`pipeline/config.py`).

| Model | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage@10 | Cold-user NDCG@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0001 | 0.0006 | 0.0003 | 0.0006 | 0.0025 | 0.0000 |
| item_knn_cf | 0.0161 | 0.1611 | 0.0963 | 0.1611 | 0.3549 | 0.2037 |
| mf_svd | 0.0006 | 0.0063 | 0.0033 | 0.0063 | 0.2179 | 0.0018 |
| mf_als_implicit | 0.0162 | 0.1621 | 0.0858 | 0.1621 | 0.3822 | 0.1531 |
| content_embeddings | 0.0021 | 0.0206 | 0.0105 | 0.0206 | 0.1882 | 0.0665 |
| ncf | 0.0111 | 0.1108 | 0.0591 | 0.1108 | 0.8047 | 0.0994 |
| two_tower | 0.0077 | 0.0773 | 0.0398 | 0.0773 | 0.9918 | 0.1142 |
| **hybrid (deployed)** | 0.0079 | 0.0792 | 0.0410 | 0.0792 | 0.9473 | 0.0796 |

Two honest takeaways worth stating rather than hiding:

1. **Classic CF (item-kNN, ALS) beats the deep models on raw ranking
   accuracy here.** With only 5–10 epochs of training and no hyperparameter
   search, NCF and the two-tower model don't out-rank a well-tuned item-kNN
   on this single, fairly small dataset — a realistic result, not a
   contrived win for the "flagship" models.
2. **But look at Coverage@10**: popularity recommends the same ~0.25% of
   the catalog to everyone; item-kNN/ALS cover ~35–38%; the two-tower and
   hybrid models cover 95–99% of the catalog. The deep models trade some
   raw accuracy for dramatically more diverse, less popularity-biased
   recommendations — a real accuracy/diversity tradeoff, and the reason a
   production system might still prefer them despite the lower NDCG.

The hybrid improves over pure two-tower on both overall NDCG (0.0410 vs.
0.0398) and Recall (0.0792 vs. 0.0773) — the content fallback for the
bottom 5% of items by rating count nets out as a small but real win, even
though the content-only model is much weaker in aggregate.

## Running locally

**One command** (after generating artifacts once, see below):

```bash
docker compose up --build
```
Frontend at http://localhost:3000, API at http://localhost:8000/docs.

### Generating the artifacts the app needs

The pipeline downloads real data and trains real models — this takes a
while (minutes on CPU, faster with a GPU) and only needs to be done once
(or whenever you want to retrain):

```bash
docker compose run --rm pipeline
```

This runs `python -m pipeline.cli run-all`: download → preprocess → train
all 8 models → evaluate → export into `artifacts/`, which `backend` then
loads at startup.

### Running without Docker

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m pipeline.cli run-all

pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000

cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

### Tests

```bash
pip install -r requirements.txt
pytest tests/
```

## Deployment

- **Backend → Render (free Web Service, Docker).** Create a Web Service
  from this repo, root directory `.`, Dockerfile path `backend/Dockerfile`
  (see `backend/README.md` for the full steps). Set
  `CORS_ALLOWED_ORIGINS` to your Vercel URL. No credit card required.
- **Frontend → Vercel.** Import the `frontend/` directory as a project, set
  `NEXT_PUBLIC_API_BASE_URL` to your Render service's URL (e.g.
  `https://your-backend.onrender.com`).

> **Note on hosting choice:** the original plan targeted Hugging Face
> Spaces' Docker SDK, but HF changed policy in mid-2026 so creating a
> Docker (or Gradio) SDK Space now requires a paid PRO plan — it's no
> longer available on the free tier. Render's free Web Service tier covers
> the same need (deploy a plain Dockerfile, no card required) and this
> backend's ~150MB memory footprint comfortably fits its 512MB limit.

Both tiers are free. Render's free tier sleeps after 15 minutes idle, so a
request after a gap can take 30–50s to wake up — the taste-profile page
shows a "waking up the model server…" message during that window rather
than looking stuck.

## Repository layout

```
pipeline/    offline ML package: data prep, 8 models, evaluation, artifact export
backend/     FastAPI service (no database, artifacts loaded at startup)
frontend/    Next.js 16 app
artifacts/   generated by the pipeline, consumed by the backend
tests/       pytest: pipeline logic + backend API
```
