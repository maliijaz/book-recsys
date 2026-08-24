"""Builds a tiny synthetic artifacts directory so backend tests never need
the real (multi-GB) pipeline output -- just enough books/embeddings to
exercise every endpoint's logic."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

N_ITEMS = 6
EMBED_DIM = 8


@pytest.fixture
def artifacts_dir(tmp_path):
    books = pd.DataFrame({
        "item_idx": list(range(N_ITEMS)),
        "book_id": [100 + i for i in range(N_ITEMS)],
        "title": ["Dune", "Dune Messiah", "Foundation", "The Hobbit", "Emma", "Pride and Prejudice"],
        "authors": ["Frank Herbert", "Frank Herbert", "Isaac Asimov", "J.R.R. Tolkien", "Jane Austen", "Jane Austen"],
        "original_publication_year": [1965.0, 1969.0, 1951.0, 1937.0, 1815.0, 1813.0],
        "average_rating": [4.25, 3.9, 4.2, 4.3, 4.0, 4.3],
        "ratings_count": [500, 200, 400, 600, 150, 700],
        "image_url": [None] * N_ITEMS,
        "small_image_url": [None] * N_ITEMS,
    })
    books.to_parquet(tmp_path / "book_metadata.parquet", index=False)

    rng = np.random.default_rng(0)
    # Cluster items 0/1 (Dune books) and 4/5 (Austen books) close together in
    # embedding space so similarity/live-rerank behavior is predictable.
    base = rng.normal(size=(N_ITEMS, EMBED_DIM)).astype(np.float32)
    base[1] = base[0] + rng.normal(scale=0.01, size=EMBED_DIM)
    base[5] = base[4] + rng.normal(scale=0.01, size=EMBED_DIM)
    norm = base / np.linalg.norm(base, axis=1, keepdims=True)
    np.save(tmp_path / "item_embeddings.npy", norm)
    np.save(tmp_path / "content_embeddings.npy", norm)  # reuse for simplicity in tests

    np.save(tmp_path / "cf_weight.npy", np.ones(N_ITEMS, dtype=np.float32))

    personas = [{
        "persona_id": 1,
        "favorite_books": [{"item_idx": 0, "book_id": 100, "title": "Dune"}],
        "recommendations": [
            {"item_idx": 1, "book_id": 101, "title": "Dune Messiah"},
            {"item_idx": 2, "book_id": 102, "title": "Foundation"},
        ],
    }]
    (tmp_path / "persona_recommendations.json").write_text(json.dumps(personas))

    metrics = [{"model": "hybrid", "overall": {"ndcg": {"10": 0.42}}, "cold_start": {}}]
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))

    return tmp_path


@pytest.fixture
def client(artifacts_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

    for mod in list(sys.modules):
        if mod == "app" or mod.startswith("app."):
            del sys.modules[mod]

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
