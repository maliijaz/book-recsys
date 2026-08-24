"""Implicit-feedback matrix factorization via the `implicit` library's ALS.

Ratings are converted to confidence weights (Hu et al. implicit-feedback
formulation: confidence = 1 + alpha * rating) rather than treated as
explicit scores, showcasing the explicit-vs-implicit technique split
alongside the SVD model.
"""
from __future__ import annotations

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from pipeline.config import RANDOM_SEED
from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender

CONFIDENCE_ALPHA = 4.0


class ALSRecommender(Recommender):
    name = "mf_als_implicit"

    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.05) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self._model: AlternatingLeastSquares | None = None

    def fit(self, dataset: Dataset) -> "ALSRecommender":
        ratings = dataset.ratings_train
        confidence = 1.0 + CONFIDENCE_ALPHA * ratings["rating"].to_numpy()
        user_items = csr_matrix(
            (confidence, (ratings["user_idx"], ratings["item_idx"])),
            shape=(dataset.n_users, dataset.n_items),
        )

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=RANDOM_SEED,
        )
        self._model.fit(user_items)
        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self._model is not None, "call fit() first"
        user_vec = self._model.user_factors[user_idx]
        return self._model.item_factors @ user_vec
