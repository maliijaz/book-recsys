"""Optional secondary dataset: Book-Crossing, used only for a severe cold-start
comparison slice (not for training the deployed product's models).

Book-Crossing requires a manual download (Kaggle auth is needed for the
`somnambwl/bookcrossing-dataset` mirror -- see README for instructions).
Place BX_Users.csv, BX-Books.csv, BX-Ratings.csv under
data/raw/bookcrossing/ before running this module.

Ratings are matched into the goodbooks-10k item space via ISBN, since the
two datasets don't share user or book ids -- this gives an independent,
much sparser signal for the same catalog, ideal for stress-testing
cold-start behavior.
"""
from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import DATA_RAW_DIR
from pipeline.data.preprocessing import Dataset

logger = logging.getLogger(__name__)

BOOKCROSSING_DIR = DATA_RAW_DIR / "bookcrossing"


def load_bookcrossing_relevant(dataset: Dataset) -> dict[int, set[int]]:
    """Return {pseudo_user_idx: {item_idx, ...}} for BX ratings matched by ISBN.

    Book-Crossing user ids are kept as their own negative-offset namespace
    (pseudo_user_idx = -bx_user_id) so they never collide with goodbooks-10k
    user_idx values -- this dict is only ever used as a standalone `relevant`
    argument to the evaluation metrics, never mixed with goodbooks train data.
    """
    ratings_path = BOOKCROSSING_DIR / "BX-Ratings.csv"
    books_path = BOOKCROSSING_DIR / "BX-Books.csv"
    if not ratings_path.exists() or not books_path.exists():
        raise FileNotFoundError(
            f"Book-Crossing CSVs not found under {BOOKCROSSING_DIR}. "
            "Download from https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset "
            "and place BX_Users.csv, BX-Books.csv, BX-Ratings.csv there."
        )

    bx_ratings = pd.read_csv(ratings_path, sep=";", encoding="latin-1", on_bad_lines="skip")
    bx_books = pd.read_csv(books_path, sep=";", encoding="latin-1", on_bad_lines="skip")

    isbn_col = next(c for c in bx_books.columns if c.upper() == "ISBN")
    bx_isbn_to_row = bx_books.set_index(isbn_col)

    goodbooks_isbns = dataset.books.reset_index()[["item_idx", "isbn"]].dropna(subset=["isbn"])
    isbn_to_item_idx = dict(zip(goodbooks_isbns["isbn"].astype(str), goodbooks_isbns["item_idx"]))

    user_col = next(c for c in bx_ratings.columns if "user" in c.lower())
    rating_isbn_col = next(c for c in bx_ratings.columns if c.upper() == "ISBN")
    rating_col = next(c for c in bx_ratings.columns if "rating" in c.lower())

    matched = bx_ratings[bx_ratings[rating_isbn_col].astype(str).isin(isbn_to_item_idx)].copy()
    matched = matched[matched[rating_col] > 0]  # 0 = implicit "read but not rated" in BX
    matched["item_idx"] = matched[rating_isbn_col].astype(str).map(isbn_to_item_idx)
    matched["pseudo_user_idx"] = -matched[user_col].astype(int)

    logger.info(
        "Matched %d/%d Book-Crossing ratings into the goodbooks-10k catalog by ISBN",
        len(matched), len(bx_ratings),
    )

    return matched.groupby("pseudo_user_idx")["item_idx"].apply(set).to_dict()
