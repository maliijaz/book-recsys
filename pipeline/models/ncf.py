"""Neural Collaborative Filtering (He et al., 2017): fused GMF + MLP.

Trained on implicit positives (rating >= POSITIVE_THRESHOLD) with random
negative sampling, via binary cross-entropy on a sigmoid output -- the
standard NCF training recipe.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset as TorchDataset

from pipeline.config import RANDOM_SEED
from pipeline.data.preprocessing import Dataset
from pipeline.models.base import Recommender

POSITIVE_THRESHOLD = 4
NEGATIVES_PER_POSITIVE = 4


class _InteractionDataset(TorchDataset):
    def __init__(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> None:
        self.users = torch.as_tensor(users, dtype=torch.long)
        self.items = torch.as_tensor(items, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.labels[idx]


class _NCFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, gmf_dim: int = 32, mlp_dim: int = 32) -> None:
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim * 2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.output = nn.Linear(gmf_dim + 16, 1)

        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        logits = self.output(torch.cat([gmf_out, mlp_out], dim=-1)).squeeze(-1)
        return logits

    def score_all_items(self, user_idx: int, n_items: int, device: torch.device) -> torch.Tensor:
        user = torch.full((n_items,), user_idx, dtype=torch.long, device=device)
        items = torch.arange(n_items, dtype=torch.long, device=device)
        with torch.no_grad():
            return torch.sigmoid(self.forward(user, items))


def _build_training_pairs(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = dataset.ratings_train[dataset.ratings_train["rating"] >= POSITIVE_THRESHOLD]
    rng = np.random.default_rng(RANDOM_SEED)

    seen_by_user = dataset.ratings_train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    users, items, labels = [], [], []
    for user_idx, item_idx in zip(positives["user_idx"], positives["item_idx"]):
        users.append(user_idx)
        items.append(item_idx)
        labels.append(1.0)
        seen = seen_by_user.get(user_idx, set())
        n_added = 0
        while n_added < NEGATIVES_PER_POSITIVE:
            candidate = rng.integers(0, dataset.n_items)
            if candidate not in seen:
                users.append(user_idx)
                items.append(candidate)
                labels.append(0.0)
                n_added += 1

    return np.array(users), np.array(items), np.array(labels, dtype=np.float32)


class NCFRecommender(Recommender):
    name = "ncf"

    def __init__(self, epochs: int = 5, batch_size: int = 1024, lr: float = 1e-3) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._net: _NCFNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._n_items = 0

    def fit(self, dataset: Dataset) -> "NCFRecommender":
        torch.manual_seed(RANDOM_SEED)
        users, items, labels = _build_training_pairs(dataset)
        loader = DataLoader(
            _InteractionDataset(users, items, labels), batch_size=self.batch_size, shuffle=True
        )

        self._net = _NCFNet(dataset.n_users, dataset.n_items).to(self._device)
        self._n_items = dataset.n_items
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self._net.train()
        for _ in range(self.epochs):
            for batch_users, batch_items, batch_labels in loader:
                batch_users = batch_users.to(self._device)
                batch_items = batch_items.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()
                logits = self._net(batch_users, batch_items)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

        self._net.eval()
        return self

    def score(self, user_idx: int) -> np.ndarray:
        assert self._net is not None, "call fit() first"
        return self._net.score_all_items(user_idx, self._n_items, self._device).cpu().numpy()
