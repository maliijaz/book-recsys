"""Item-based k-NN collaborative filtering.

Precomputes the top-N most similar items for every item (cosine similarity
over the item-user rating matrix), then scores candidate items for a user as
a similarity-weighted sum over the items that user has already rated.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender


class ItemKNNRecommender(Recommender):
    name = "item_knn_cf"

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors
        self._neighbors: dict[int, list[tuple[int, float]]] = {}
        self._user_ratings: dict[int, list[tuple[int, float]]] = {}

    def fit(self, dataset: Dataset) -> "ItemKNNRecommender":
        ratings = dataset.ratings_train
        item_user = csr_matrix(
            (ratings["rating"], (ratings["item_idx"], ratings["user_idx"])),
            shape=(dataset.n_items, dataset.n_users),
        )

        n_neighbors = min(self.n_neighbors + 1, dataset.n_items)
        model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        model.fit(item_user)
        distances, indices = model.kneighbors(item_user)

        self._neighbors = {}
        for item_idx in range(dataset.n_items):
            neighbor_ids = indices[item_idx]
            similarities = 1.0 - distances[item_idx]
            pairs = [
                (int(nid), float(sim))
                for nid, sim in zip(neighbor_ids, similarities)
                if nid != item_idx
            ]
            self._neighbors[item_idx] = pairs[: self.n_neighbors]

        self._user_ratings = (
            ratings.groupby("user_idx")
            .apply(lambda g: list(zip(g["item_idx"], g["rating"])), include_groups=False)
            .to_dict()
        )
        self._n_items = dataset.n_items
        return self

    def score(self, user_idx: int) -> np.ndarray:
        scores = np.zeros(self._n_items)
        for item_idx, rating in self._user_ratings.get(user_idx, []):
            for neighbor_idx, similarity in self._neighbors.get(item_idx, []):
                scores[neighbor_idx] += similarity * rating
        return scores
