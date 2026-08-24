"""Export everything the backend needs, as plain files it loads at startup.

No database: metadata goes to parquet, embeddings to .npy, precomputed
persona recommendations and offline metrics to .json. The backend never
imports the pipeline's heavy ML dependencies (torch, sentence-transformers,
surprise, implicit) -- it only does numpy dot products against these
already-trained artifacts.
"""
from __future__ import annotations

import json

import numpy as np

from pipeline.config import ARTIFACTS_DIR
from pipeline.data.preprocessing import Dataset
from pipeline.models.hybrid import HybridRecommender

N_PERSONAS = 20
PERSONA_TOP_K = 10
PERSONA_MIN_RATINGS = 20
PERSONA_MAX_RATINGS = 300


def export_book_metadata(dataset: Dataset) -> None:
    books = dataset.books.sort_index().copy()
    books = books.reset_index().rename(columns={"index": "item_idx"})
    keep_cols = [
        "item_idx", "book_id", "title", "authors", "original_publication_year",
        "average_rating", "ratings_count", "image_url", "small_image_url",
    ]
    books[keep_cols].to_parquet(ARTIFACTS_DIR / "book_metadata.parquet", index=False)


def export_embeddings(item_embeddings: np.ndarray, content_embeddings: np.ndarray) -> None:
    np.save(ARTIFACTS_DIR / "item_embeddings.npy", item_embeddings.astype(np.float32))
    np.save(ARTIFACTS_DIR / "content_embeddings.npy", content_embeddings.astype(np.float32))


def export_cf_weight(cf_weight: np.ndarray) -> None:
    np.save(ARTIFACTS_DIR / "cf_weight.npy", cf_weight.astype(np.float32))


def export_personas(dataset: Dataset, hybrid: HybridRecommender, rng_seed: int = 42) -> None:
    train_user_counts = dataset.ratings_train["user_idx"].value_counts()
    eligible = train_user_counts[
        (train_user_counts >= PERSONA_MIN_RATINGS) & (train_user_counts <= PERSONA_MAX_RATINGS)
    ].index.to_numpy()

    rng = np.random.default_rng(rng_seed)
    persona_users = rng.choice(eligible, size=min(N_PERSONAS, len(eligible)), replace=False)

    train_by_user = dataset.ratings_train.groupby("user_idx")
    book_titles = dataset.books["title"]

    personas = []
    for user_idx in persona_users:
        user_idx = int(user_idx)
        recs = hybrid.recommend_all(dataset, [user_idx], k=PERSONA_TOP_K)[user_idx]
        top_rated = (
            train_by_user.get_group(user_idx).sort_values("rating", ascending=False).head(5)
        )
        personas.append({
            "persona_id": user_idx,
            "favorite_books": [
                {"item_idx": int(i), "book_id": dataset.item_idx_to_id[int(i)], "title": str(book_titles.get(int(i), ""))}
                for i in top_rated["item_idx"]
            ],
            "recommendations": [
                {"item_idx": int(i), "book_id": dataset.item_idx_to_id[int(i)], "title": str(book_titles.get(int(i), ""))}
                for i in recs
            ],
        })

    (ARTIFACTS_DIR / "persona_recommendations.json").write_text(json.dumps(personas, indent=2))


def _serialize_metrics(model_results: dict) -> dict:
    def _k_keyed(d: dict[int, float]) -> dict[str, float]:
        return {str(k): round(v, 4) for k, v in d.items()}

    return {
        "model": model_results["model"],
        "overall": {metric: _k_keyed(vals) for metric, vals in model_results["overall"].items()},
        "cold_start": {
            "cold_user_metrics": {
                metric: _k_keyed(vals) for metric, vals in model_results["cold_start"]["cold_user_metrics"].items()
            },
            "cold_item_recall": _k_keyed(model_results["cold_start"]["cold_item_recall"]),
            "n_cold_users": model_results["cold_start"]["n_cold_users"],
            "n_cold_items": model_results["cold_start"]["n_cold_items"],
        },
    }


def export_metrics(all_model_results: list[dict]) -> None:
    serialized = [_serialize_metrics(r) for r in all_model_results]
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(serialized, indent=2))
