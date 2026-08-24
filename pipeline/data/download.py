"""Download the goodbooks-10k dataset from its public GitHub mirror.

goodbooks-10k is freely downloadable with no access gating:
https://github.com/zygmuntz/goodbooks-10k
"""
import logging

import requests

from pipeline.config import DATA_RAW_DIR, GOODBOOKS_BASE_URL, GOODBOOKS_FILES

logger = logging.getLogger(__name__)


def download_goodbooks(force: bool = False) -> None:
    """Download all goodbooks-10k CSVs into data/raw/ if not already present."""
    for filename in GOODBOOKS_FILES:
        dest = DATA_RAW_DIR / filename
        if dest.exists() and not force:
            logger.info("Skipping %s (already downloaded)", filename)
            continue

        url = f"{GOODBOOKS_BASE_URL}/{filename}"
        logger.info("Downloading %s -> %s", url, dest)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)

    logger.info("goodbooks-10k download complete: %s", DATA_RAW_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_goodbooks()
