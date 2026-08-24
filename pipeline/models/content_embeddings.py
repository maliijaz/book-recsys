"""Content-based recommender: sentence-transformer embeddings of title/author/tags.

Handles cold-start items with zero interactions (a brand-new book still has
a title and tags, so it gets a usable embedding immediately) and powers the
`/books/{id}/similar` endpoint via item-item cosine similarity. Its output
embedding matrix is also reused by the hybrid model.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender

MODEL_NAME = "all-MiniLM-L6-v2"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class ContentEmbeddingRecommender(Recommender):
    name = "content_embeddings"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.item_embeddings: np.ndarray | None = None
        self._user_ratings: dict[int, list[tuple[int, float]]] = {}

    def fit(self, dataset: Dataset) -> "ContentEmbeddingRecommender":
        model = SentenceTransformer(self.model_name)
        texts = dataset.books.sort_index()["content_text"].fillna("").tolist()
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        self.item_embeddings = _normalize(embeddings.astype(np.float32))

        self._user_ratings = (
            dataset.ratings_train.groupby("user_idx")
            .apply(lambda g: list(zip(g["item_idx"], g["rating"])), include_groups=False)
            .to_dict()
        )
        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self.item_embeddings is not None, "call fit() first"
        pairs = self._user_ratings.get(user_idx, [])
        if not pairs:
            return np.zeros(self.item_embeddings.shape[0])
        item_idxs, ratings = zip(*pairs)
        weights = np.array(ratings, dtype=np.float32)
        weights = weights / weights.sum()
        profile = (self.item_embeddings[list(item_idxs)] * weights[:, None]).sum(axis=0)
        return self.item_embeddings @ profile

    def similar_items(self, item_idx: int, k: int = 10) -> list[int]:
        assert self.item_embeddings is not None, "call fit() first"
        sims = self.item_embeddings @ self.item_embeddings[item_idx]
        sims[item_idx] = -np.inf
        top_idx = np.argpartition(-sims, k)[:k]
        return top_idx[np.argsort(-sims[top_idx])].tolist()
