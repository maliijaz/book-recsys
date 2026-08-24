"""Explicit-feedback matrix factorization via scikit-surprise's SVD.

scikit-surprise is in bugfix-only maintenance mode upstream but still
receives releases (1.1.5, May 2026) -- stable enough to pin and use as-is.
After fitting we pull the raw factors/biases out into plain numpy arrays so
scoring the full catalog for a user is a single vectorized dot product
instead of thousands of per-item `predict()` calls.
"""
from __future__ import annotations

import numpy as np
from surprise import SVD, Dataset as SurpriseDataset, Reader

from pipeline.config import RANDOM_SEED
from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender


class SVDRecommender(Recommender):
    name = "mf_svd"

    def __init__(self, n_factors: int = 64, n_epochs: int = 20) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._user_bias: np.ndarray | None = None
        self._item_bias: np.ndarray | None = None
        self._global_mean = 0.0

    def fit(self, dataset: Dataset) -> "SVDRecommender":
        reader = Reader(rating_scale=(1, 5))
        surprise_data = SurpriseDataset.load_from_df(
            dataset.ratings_train[["user_idx", "item_idx", "rating"]], reader
        )
        trainset = surprise_data.build_full_trainset()

        algo = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, random_state=RANDOM_SEED)
        algo.fit(trainset)

        self._global_mean = trainset.global_mean
        self._user_factors = np.zeros((dataset.n_users, self.n_factors))
        self._item_factors = np.zeros((dataset.n_items, self.n_factors))
        self._user_bias = np.zeros(dataset.n_users)
        self._item_bias = np.zeros(dataset.n_items)

        for raw_uid in range(dataset.n_users):
            if trainset.knows_user(raw_uid):
                inner = trainset.to_inner_uid(raw_uid)
                self._user_factors[raw_uid] = algo.pu[inner]
                self._user_bias[raw_uid] = algo.bu[inner]

        for raw_iid in range(dataset.n_items):
            if trainset.knows_item(raw_iid):
                inner = trainset.to_inner_iid(raw_iid)
                self._item_factors[raw_iid] = algo.qi[inner]
                self._item_bias[raw_iid] = algo.bi[inner]

        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self._item_factors is not None, "call fit() first"
        return (
            self._global_mean
            + self._user_bias[user_idx]
            + self._item_bias
            + self._item_factors @ self._user_factors[user_idx]
        )
