"""Two-tower embedding retrieval model -- the flagship deep-learning model.

Item tower = learned id embedding + a projection of the precomputed content
embedding (title/author/tags), so even a zero-interaction item still gets a
content-grounded vector. User tower = learned id embedding for known users.
Trained with in-batch sampled softmax (each batch's positives double as
everyone else's negatives), the standard large-scale two-tower recipe.

The exported item-embedding matrix plus `pool_liked_items` (mean-pooling a
handful of liked items' embeddings into an ad-hoc user vector) are exactly
what the backend's `/recommendations/live` endpoint needs: no login, no
retraining, just a forward pass through embeddings that already exist.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from pipeline.config import N_ITEM_EMBEDDING_DIM, RANDOM_SEED
from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender

POSITIVE_THRESHOLD = 4


class _PositivePairsDataset(TorchDataset):
    def __init__(self, users: np.ndarray, items: np.ndarray) -> None:
        self.users = torch.as_tensor(users, dtype=torch.long)
        self.items = torch.as_tensor(items, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx]


class _ItemTower(nn.Module):
    def __init__(self, n_items: int, content_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(n_items, embedding_dim)
        self.content_proj = nn.Linear(content_dim, embedding_dim)
        nn.init.normal_(self.id_embedding.weight, std=0.01)

    def forward(self, item_idx: torch.Tensor, content_embeddings: torch.Tensor) -> torch.Tensor:
        return self.id_embedding(item_idx) + self.content_proj(content_embeddings[item_idx])

    def all_embeddings(self, content_embeddings: torch.Tensor) -> torch.Tensor:
        all_ids = torch.arange(self.id_embedding.num_embeddings, device=content_embeddings.device)
        return self.id_embedding(all_ids) + self.content_proj(content_embeddings)


class _UserTower(nn.Module):
    def __init__(self, n_users: int, embedding_dim: int) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(n_users, embedding_dim)
        nn.init.normal_(self.id_embedding.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.id_embedding(user_idx)


def pool_liked_items(item_embeddings: np.ndarray, liked_item_idxs: list[int]) -> np.ndarray:
    """Mean-pool liked items' embeddings into an ad-hoc user vector.

    This is the cold/anonymous-user proxy used both offline (cold-start
    eval) and online (the live re-ranking endpoint) -- no user id needed.
    """
    if not liked_item_idxs:
        return np.zeros(item_embeddings.shape[1], dtype=item_embeddings.dtype)
    return item_embeddings[liked_item_idxs].mean(axis=0)


class TwoTowerRecommender(Recommender):
    name = "two_tower"

    def __init__(
        self,
        embedding_dim: int = N_ITEM_EMBEDDING_DIM,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-3,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embeddings: np.ndarray | None = None
        self.user_embeddings: np.ndarray | None = None

    def fit(self, dataset: Dataset, content_embeddings: np.ndarray) -> "TwoTowerRecommender":
        torch.manual_seed(RANDOM_SEED)
        positives = dataset.ratings_train[dataset.ratings_train["rating"] >= POSITIVE_THRESHOLD]
        loader = DataLoader(
            _PositivePairsDataset(positives["user_idx"].to_numpy(), positives["item_idx"].to_numpy()),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        content_t = torch.as_tensor(content_embeddings, dtype=torch.float32, device=self._device)
        item_tower = _ItemTower(dataset.n_items, content_embeddings.shape[1], self.embedding_dim).to(self._device)
        user_tower = _UserTower(dataset.n_users, self.embedding_dim).to(self._device)

        optimizer = torch.optim.Adam(
            list(item_tower.parameters()) + list(user_tower.parameters()), lr=self.lr
        )

        item_tower.train()
        user_tower.train()
        for _ in range(self.epochs):
            for batch_users, batch_items in loader:
                batch_users = batch_users.to(self._device)
                batch_items = batch_items.to(self._device)

                user_vecs = user_tower(batch_users)
                item_vecs = item_tower(batch_items, content_t)

                logits = user_vecs @ item_vecs.T  # in-batch sampled softmax
                labels = torch.arange(logits.shape[0], device=self._device)
                loss = nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        item_tower.eval()
        user_tower.eval()
        with torch.no_grad():
            self.item_embeddings = item_tower.all_embeddings(content_t).cpu().numpy()
            all_user_ids = torch.arange(dataset.n_users, device=self._device)
            self.user_embeddings = user_tower(all_user_ids).cpu().numpy()

        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self.item_embeddings is not None and self.user_embeddings is not None
        return self.item_embeddings @ self.user_embeddings[user_idx]

    def score_for_liked_items(self, liked_item_idxs: list[int]) -> np.ndarray:
        """Live re-ranking path: score all items against a pooled ad-hoc user vector."""
        assert self.item_embeddings is not None
        pooled = pool_liked_items(self.item_embeddings, liked_item_idxs)
        return self.item_embeddings @ pooled
