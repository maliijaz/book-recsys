# Notebooks

Exploratory / EDA scratch space only. Nothing in `pipeline/`, `backend/`, or
`frontend/` depends on anything produced here — the real, reproducible
pipeline lives in `pipeline/` and runs via `python -m pipeline.cli run-all`.

Suggested notebook to add if you want to explore the data interactively:

- `01_eda.ipynb` — ratings distribution, sparsity, genre/tag coverage over
  `data/raw/*.csv` (run `python -m pipeline.data.download` first).
