"""Shared paths and constants for the offline ML pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
# Lives inside backend/ (not repo root) so the backend directory is fully
# self-contained -- required for deploying it as a standalone Vercel
# project (Root Directory = backend/) with zero extra bundling config.
ARTIFACTS_DIR = ROOT_DIR / "backend" / "artifacts"

GOODBOOKS_BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
GOODBOOKS_FILES = ["ratings.csv", "books.csv", "book_tags.csv", "tags.csv", "to_read.csv"]

BOOKCROSSING_KAGGLE_DATASET = "somnambwl/bookcrossing-dataset"
BOOKCROSSING_FILES = ["BX_Users.csv", "BX-Books.csv", "BX-Ratings.csv"]

N_ITEM_EMBEDDING_DIM = 64
N_CONTENT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

RANDOM_SEED = 42

# Users/items with fewer than this many training interactions are treated
# as "cold" for the cold-start evaluation slice and the hybrid model's
# CF/content blend weight. goodbooks-10k is a curated "most-rated" subset --
# every item has >= 8 and every user has >= 18 training ratings, so a naive
# threshold like 5 produces an empty cold-start slice. These thresholds are
# picked from the actual train-set distribution to isolate a meaningful
# bottom slice (~1% of users, ~5% of items) instead of a hard interaction-
# count floor.
COLD_USER_THRESHOLD = 40
COLD_ITEM_THRESHOLD = 100

TOP_K_VALUES = (5, 10, 20)

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
