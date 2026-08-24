"""Hybrid model: two-tower CF score blended with content-embedding similarity.

Per-item blend weight favors CF for well-observed items and falls back to
pure content similarity for items in `dataset.cold_item_idxs` -- directly
targeting collaborative filtering's main weakness. This is the model the
backend deploys.
"""
from __future__ import annotations

import numpy as np

from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender
from pipeline.models.content_embeddings import ContentEmbeddingRecommender
from pipeline.models.two_tower import TwoTowerRecommender, pool_liked_items


def _z_normalize(scores: np.ndarray) -> np.ndarray:
    std = scores.std()
    if std == 0:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / std


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(self, two_tower: TwoTowerRecommender, content: ContentEmbeddingRecommender) -> None:
        self.two_tower = two_tower
        self.content = content
        self.cf_weight: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "HybridRecommender":
        # Both sub-models must already be fit; this just derives the per-item blend weight,
        # reusing the same cold-item definition the evaluation slice uses.
        weights = np.ones(dataset.n_items)
        weights[list(dataset.cold_item_idxs)] = 0.0
        self.cf_weight = weights
        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self.cf_weight is not None, "call fit() first"
        cf_scores = _z_normalize(self.two_tower.score(user_idx))
        content_scores = _z_normalize(self.content.score(user_idx))
        return self.cf_weight * cf_scores + (1 - self.cf_weight) * content_scores

    def score_for_liked_items(self, liked_item_idxs: list[int]) -> np.ndarray:
        """Live re-ranking path used by the backend: no user_idx required."""
        assert self.cf_weight is not None and self.two_tower.item_embeddings is not None
        item_embeddings = self.two_tower.item_embeddings
        content_embeddings = self.content.item_embeddings
        assert content_embeddings is not None

        pooled_cf = pool_liked_items(item_embeddings, liked_item_idxs)
        cf_scores = _z_normalize(item_embeddings @ pooled_cf)

        pooled_content = pool_liked_items(content_embeddings, liked_item_idxs)
        content_scores = _z_normalize(content_embeddings @ pooled_content)

        return self.cf_weight * cf_scores + (1 - self.cf_weight) * content_scores
