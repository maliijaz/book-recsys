"""Shared recommender interface: every model exposes fit() + score(user_idx).

Ranking/exclusion logic (exclude already-seen items, take top-K) is
implemented once here so each model only needs to produce a per-item score
vector.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from pipeline.data.preprocessing import Dataset


class Recommender:
    name: str = "base"

    def fit(self, dataset: Dataset) -> "Recommender":
        raise NotImplementedError

    def score(self, user_idx: int) -> np.ndarray:
        """Return a (n_items,) array of scores for every item, higher = better."""
        raise NotImplementedError

    def recommend_all(
        self,
        dataset: Dataset,
        user_idxs: Iterable[int],
        k: int,
        exclude_seen: bool = True,
    ) -> dict[int, list[int]]:
        train_user_items = dataset.train_user_items()
        out: dict[int, list[int]] = {}
        for u in user_idxs:
            scores = self.score(u).astype(np.float64).copy()
            excluded = False
            if exclude_seen:
                seen = train_user_items.get(u)
                if seen:
                    scores[list(seen)] = -np.inf
                    excluded = True
            n_candidates = min(k, scores.shape[0])
            top_idx = np.argpartition(-scores, n_candidates - 1)[:n_candidates]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            if excluded:
                # -inf always sorts last, so this only drops seen items that
                # spilled into the top-k because too few unseen items remained.
                top_idx = top_idx[scores[top_idx] != -np.inf]
            out[u] = top_idx.tolist()
        return out
