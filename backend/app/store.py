"""Loads precomputed artifacts once at process start and serves every read
from in-memory numpy/pandas structures -- no database, no per-request I/O,
no model inference. This is the entire "batch-precomputed" half of the
serving design; `live_recommendations` below is the "live re-ranking" half,
and it's still pure numpy, just computed on demand from a request's liked
book ids instead of at pipeline export time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _z_normalize(scores: np.ndarray) -> np.ndarray:
    std = scores.std()
    if std == 0:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std


class ArtifactNotFoundError(RuntimeError):
    pass


class Store:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self._load()

    def _load(self) -> None:
        required = [
            "book_metadata.parquet", "item_embeddings.npy", "content_embeddings.npy",
            "cf_weight.npy", "persona_recommendations.json", "metrics.json",
        ]
        missing = [f for f in required if not (self.artifacts_dir / f).exists()]
        if missing:
            raise ArtifactNotFoundError(
                f"Missing artifacts in {self.artifacts_dir}: {missing}. "
                "Run `python -m pipeline.cli run-all` to generate them."
            )

        self.books_df = (
            pd.read_parquet(self.artifacts_dir / "book_metadata.parquet")
            .set_index("item_idx")
            .sort_index()
        )
        self.book_id_to_item_idx: dict[int, int] = dict(
            zip(self.books_df["book_id"].astype(int), self.books_df.index)
        )
        self.item_embeddings = np.load(self.artifacts_dir / "item_embeddings.npy")
        self.content_embeddings = np.load(self.artifacts_dir / "content_embeddings.npy")
        self.cf_weight = np.load(self.artifacts_dir / "cf_weight.npy")
        self.personas: list[dict] = json.loads(
            (self.artifacts_dir / "persona_recommendations.json").read_text()
        )
        self.personas_by_id = {p["persona_id"]: p for p in self.personas}
        self.metrics: list[dict] = json.loads((self.artifacts_dir / "metrics.json").read_text())

    # -- books -----------------------------------------------------------

    def _book_summary(self, item_idx: int) -> dict:
        row = self.books_df.loc[item_idx]
        return {
            "book_id": int(row["book_id"]),
            "title": str(row["title"]),
            "authors": str(row["authors"]),
            "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
            "original_publication_year": (
                float(row["original_publication_year"])
                if pd.notna(row["original_publication_year"]) else None
            ),
            "image_url": row.get("image_url") if pd.notna(row.get("image_url")) else None,
        }

    def _summaries_for_idxs(self, idxs) -> list[dict]:
        return [self._book_summary(int(i)) for i in idxs]

    def list_books(self, query: str | None, page: int, page_size: int) -> tuple[list[dict], int]:
        df = self.books_df
        if query:
            mask = (
                df["title"].str.contains(query, case=False, na=False)
                | df["authors"].str.contains(query, case=False, na=False)
            )
            df = df[mask]
        total = len(df)
        start = (page - 1) * page_size
        page_idxs = df.index[start:start + page_size]
        return self._summaries_for_idxs(page_idxs), total

    def get_book_detail(self, book_id: int) -> dict | None:
        item_idx = self.book_id_to_item_idx.get(book_id)
        if item_idx is None:
            return None
        row = self.books_df.loc[item_idx]
        summary = self._book_summary(item_idx)
        summary["ratings_count"] = int(row["ratings_count"]) if pd.notna(row["ratings_count"]) else None
        summary["small_image_url"] = row.get("small_image_url") if pd.notna(row.get("small_image_url")) else None
        return summary

    def similar_books(self, book_id: int, k: int = 10) -> list[dict] | None:
        item_idx = self.book_id_to_item_idx.get(book_id)
        if item_idx is None:
            return None
        sims = self.content_embeddings @ self.content_embeddings[item_idx]
        sims = sims.copy()
        sims[item_idx] = -np.inf
        n = min(k, sims.shape[0] - 1)
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        return self._summaries_for_idxs(top)

    # -- personas (precomputed batch path) --------------------------------

    def list_personas(self) -> list[dict]:
        return [
            {"persona_id": p["persona_id"], "favorite_books": [
                self._book_summary(b["item_idx"]) for b in p["favorite_books"]
            ]}
            for p in self.personas
        ]

    def get_persona(self, persona_id: int) -> dict | None:
        p = self.personas_by_id.get(persona_id)
        if p is None:
            return None
        return {
            "persona_id": p["persona_id"],
            "favorite_books": [self._book_summary(b["item_idx"]) for b in p["favorite_books"]],
            "recommendations": [self._book_summary(b["item_idx"]) for b in p["recommendations"]],
        }

    def get_persona_recommendations(self, persona_id: int) -> list[dict] | None:
        p = self.personas_by_id.get(persona_id)
        if p is None:
            return None
        return [self._book_summary(b["item_idx"]) for b in p["recommendations"]]

    # -- live re-ranking (no login, no persisted user) ---------------------

    def live_recommendations(self, liked_book_ids: list[int], k: int = 10) -> dict | None:
        liked_idxs = [
            self.book_id_to_item_idx[bid] for bid in liked_book_ids if bid in self.book_id_to_item_idx
        ]
        if not liked_idxs:
            return None

        pooled_cf = self.item_embeddings[liked_idxs].mean(axis=0)
        cf_scores = _z_normalize(self.item_embeddings @ pooled_cf)

        pooled_content = self.content_embeddings[liked_idxs].mean(axis=0)
        content_scores = _z_normalize(self.content_embeddings @ pooled_content)

        scores = self.cf_weight * cf_scores + (1 - self.cf_weight) * content_scores
        scores = scores.copy()
        scores[liked_idxs] = -np.inf  # never recommend back what they just picked

        n = min(k, scores.shape[0] - len(liked_idxs))
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]

        return {
            "based_on": self._summaries_for_idxs(liked_idxs),
            "recommendations": self._summaries_for_idxs(top),
        }

    # -- model card ---------------------------------------------------------

    def get_metrics(self) -> list[dict]:
        return self.metrics
