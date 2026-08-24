"""Popularity baseline: recommend the same globally popular items to everyone."""
from __future__ import annotations

import numpy as np

from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> "PopularityRecommender":
        counts = np.zeros(dataset.n_items)
        rating_sums = np.zeros(dataset.n_items)
        grouped = dataset.ratings_train.groupby("item_idx")["rating"]
        for item_idx, group in grouped:
            counts[item_idx] = len(group)
            rating_sums[item_idx] = group.sum()

        avg_rating = np.divide(
            rating_sums, counts, out=np.zeros_like(rating_sums), where=counts > 0
        )
        # Bayesian-average-style score: balances popularity (count) with quality (avg rating).
        global_mean = dataset.ratings_train["rating"].mean()
        prior_weight = 10.0
        self._scores = (counts * avg_rating + prior_weight * global_mean) / (counts + prior_weight)
        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self._scores is not None, "call fit() first"
        return self._scores
