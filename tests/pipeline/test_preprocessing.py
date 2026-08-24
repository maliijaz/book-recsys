"""Split-size and no-leakage checks for the leave-k-out preprocessing, run
against tiny synthetic CSVs so no real download is required."""
import pandas as pd
import pytest

from pipeline.data.preprocessing import load_and_split


@pytest.fixture
def synthetic_raw_dir(tmp_path, monkeypatch):
    books = pd.DataFrame({
        "book_id": [1, 2, 3, 4, 5],
        "goodreads_book_id": [101, 102, 103, 104, 105],
        "title": ["A", "B", "C", "D", "E"],
        "authors": ["Auth1", "Auth2", "Auth3", "Auth4", "Auth5"],
    })
    # user 1: only 1 rating -> must stay entirely in train (nothing to hold out)
    # user 2: 4 ratings -> 1 held out for test, 3 remain in train
    # user 3: 3 ratings -> 1 held out for test, 2 remain in train
    ratings = pd.DataFrame({
        "user_id": [1, 2, 2, 2, 2, 3, 3, 3],
        "book_id": [1, 1, 2, 3, 4, 2, 3, 5],
        "rating": [5, 4, 3, 5, 2, 4, 4, 5],
    })
    tags = pd.DataFrame({"tag_id": [1], "tag_name": ["fiction"]})
    book_tags = pd.DataFrame({"goodreads_book_id": [101], "tag_id": [1], "count": [10]})

    books.to_csv(tmp_path / "books.csv", index=False)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    tags.to_csv(tmp_path / "tags.csv", index=False)
    book_tags.to_csv(tmp_path / "book_tags.csv", index=False)

    monkeypatch.setattr("pipeline.data.preprocessing.DATA_RAW_DIR", tmp_path)
    return tmp_path


def test_split_sizes(synthetic_raw_dir):
    dataset = load_and_split(test_holdout_per_user=1)

    assert dataset.n_items == 5
    assert dataset.n_users == 3
    assert len(dataset.ratings_test) == 2  # only user 2 and user 3 qualify
    assert len(dataset.ratings_train) == 8 - 2


def test_single_rating_users_never_held_out(synthetic_raw_dir):
    dataset = load_and_split(test_holdout_per_user=1)

    user1_idx = dataset.user_id_to_idx[1]
    assert user1_idx not in set(dataset.ratings_test["user_idx"])
    assert user1_idx in set(dataset.ratings_train["user_idx"])


def test_no_leakage_between_train_and_test(synthetic_raw_dir):
    dataset = load_and_split(test_holdout_per_user=1)

    train_pairs = set(zip(dataset.ratings_train["user_idx"], dataset.ratings_train["item_idx"]))
    test_pairs = set(zip(dataset.ratings_test["user_idx"], dataset.ratings_test["item_idx"]))
    assert train_pairs.isdisjoint(test_pairs)


def test_each_eligible_user_has_exactly_one_test_row(synthetic_raw_dir):
    dataset = load_and_split(test_holdout_per_user=1)
    counts = dataset.ratings_test["user_idx"].value_counts()
    assert (counts == 1).all()


def test_books_dataframe_keeps_book_id_column(synthetic_raw_dir):
    # Regression test: an earlier version of load_and_split re-indexed the
    # books frame from book_id -> item_idx via a bare set_index(), which
    # silently drops the old index instead of keeping it as a column.
    dataset = load_and_split(test_holdout_per_user=1)
    assert "book_id" in dataset.books.columns
    for book_id, item_idx in dataset.item_id_to_idx.items():
        assert dataset.books.loc[item_idx, "book_id"] == book_id


def test_cold_item_idxs_computed_from_train_only(synthetic_raw_dir):
    dataset = load_and_split(test_holdout_per_user=1)
    # book_id=5 has exactly one rating in the whole dataset, so however the
    # random holdout falls it has at most 1 train interaction -- always cold.
    item5_idx = dataset.item_id_to_idx[5]
    assert item5_idx in dataset.cold_item_idxs
