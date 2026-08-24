"""Load goodbooks-10k, build id mappings, and construct train/test splits.

goodbooks-10k ships no timestamps on ratings.csv, so a genuine time-based
split isn't possible here. We use leave-one-out per user instead: for each
user with enough interactions, one random rating is held out as the test
positive and the rest go to train. This is the standard protocol for
implicit-feedback ranking evaluation when chronology isn't available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.config import COLD_ITEM_THRESHOLD, COLD_USER_THRESHOLD, DATA_RAW_DIR, RANDOM_SEED


@dataclass
class Dataset:
    """Preprocessed goodbooks-10k data, ready for model training/evaluation."""

    ratings_train: pd.DataFrame  # columns: user_idx, item_idx, rating
    ratings_test: pd.DataFrame  # columns: user_idx, item_idx, rating (one row per eligible user)
    books: pd.DataFrame  # indexed by item_idx, metadata columns
    n_users: int
    n_items: int
    item_id_to_idx: dict[int, int]
    item_idx_to_id: dict[int, int]
    user_id_to_idx: dict[int, int]
    cold_item_idxs: set[int] = field(default_factory=set)
    cold_user_idxs: set[int] = field(default_factory=set)

    def train_user_items(self) -> dict[int, set[int]]:
        """Map user_idx -> set of item_idx rated in train (for excluding seen items)."""
        return self.ratings_train.groupby("user_idx")["item_idx"].apply(set).to_dict()


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(DATA_RAW_DIR / "ratings.csv")
    books = pd.read_csv(DATA_RAW_DIR / "books.csv")
    tags = pd.read_csv(DATA_RAW_DIR / "tags.csv")
    book_tags = pd.read_csv(DATA_RAW_DIR / "book_tags.csv")
    return ratings, books, tags, book_tags


def _build_item_text(books: pd.DataFrame, tags: pd.DataFrame, book_tags: pd.DataFrame) -> pd.Series:
    """Concatenate title/author/top-tags into a single text field per book, for content embeddings."""
    tag_lookup = tags.set_index("tag_id")["tag_name"]
    top_tags = (
        book_tags.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
        .groupby("goodreads_book_id")
        .head(10)
    )
    top_tags = top_tags.assign(tag_name=top_tags["tag_id"].map(tag_lookup))
    tags_per_book = (
        top_tags.groupby("goodreads_book_id")["tag_name"]
        .apply(lambda names: " ".join(str(n) for n in names if pd.notna(n)))
    )

    merged = books.set_index("goodreads_book_id")
    text = (
        merged["title"].fillna("")
        + ". by "
        + merged["authors"].fillna("")
        + ". "
        + merged.index.to_series().map(tags_per_book).fillna("")
    )
    text.index = merged["book_id"].values
    return text


def load_and_split(test_holdout_per_user: int = 1) -> Dataset:
    """Load goodbooks-10k, build id mappings, and produce a leave-k-out train/test split."""
    ratings, books, tags, book_tags = _load_raw()
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"])

    item_ids = np.sort(books["book_id"].unique())
    item_id_to_idx = {int(bid): i for i, bid in enumerate(item_ids)}
    item_idx_to_id = {i: int(bid) for bid, i in item_id_to_idx.items()}

    user_ids = np.sort(ratings["user_id"].unique())
    user_id_to_idx = {int(uid): i for i, uid in enumerate(user_ids)}

    ratings = ratings[ratings["book_id"].isin(item_id_to_idx)].copy()
    ratings["item_idx"] = ratings["book_id"].map(item_id_to_idx)
    ratings["user_idx"] = ratings["user_id"].map(user_id_to_idx)

    rng = np.random.default_rng(RANDOM_SEED)

    def _split_user(group: pd.DataFrame) -> pd.Series:
        if len(group) <= test_holdout_per_user:
            return pd.Series(False, index=group.index)
        test_positions = rng.choice(group.index, size=test_holdout_per_user, replace=False)
        mask = pd.Series(False, index=group.index)
        mask.loc[test_positions] = True
        return mask

    is_test = ratings.groupby("user_idx", group_keys=False).apply(_split_user, include_groups=False)
    ratings_train = ratings.loc[~is_test, ["user_idx", "item_idx", "rating"]].reset_index(drop=True)
    ratings_test = ratings.loc[is_test, ["user_idx", "item_idx", "rating"]].reset_index(drop=True)

    item_text = _build_item_text(books, tags, book_tags)
    books_indexed = books.set_index("book_id")
    books_indexed["item_idx"] = books_indexed.index.map(item_id_to_idx)
    books_indexed = books_indexed.dropna(subset=["item_idx"])
    books_indexed["item_idx"] = books_indexed["item_idx"].astype(int)
    books_indexed["content_text"] = books_indexed.index.map(item_text)
    # set_index() below would otherwise silently drop the current index
    # (book_id) instead of keeping it as a column -- restore it first.
    books_by_idx = books_indexed.reset_index().set_index("item_idx").sort_index()

    train_item_counts = ratings_train["item_idx"].value_counts()
    train_user_counts = ratings_train["user_idx"].value_counts()
    all_item_idxs = set(range(len(item_ids)))
    all_user_idxs = set(range(len(user_ids)))
    cold_item_idxs = {
        i for i in all_item_idxs if train_item_counts.get(i, 0) < COLD_ITEM_THRESHOLD
    }
    cold_user_idxs = {
        u for u in all_user_idxs if train_user_counts.get(u, 0) < COLD_USER_THRESHOLD
    }

    return Dataset(
        ratings_train=ratings_train,
        ratings_test=ratings_test,
        books=books_by_idx,
        n_users=len(user_ids),
        n_items=len(item_ids),
        item_id_to_idx=item_id_to_idx,
        item_idx_to_id=item_idx_to_id,
        user_id_to_idx=user_id_to_idx,
        cold_item_idxs=cold_item_idxs,
        cold_user_idxs=cold_user_idxs,
    )
