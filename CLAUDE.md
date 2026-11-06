# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

End-to-end retail demand forecasting on the Kaggle M5 dataset (Walmart item-store daily sales). Each `(item_id, store_id)` pair is one time series. Current status: ingestion, data quality, feature engineering and naive/seasonal-naive baselines are done; statistical/ML models, evaluation, and Azure deployment/monitoring are planned (see README checklist).

## Commands

Uses `uv` with Python 3.13+. Run modules from the repo root with `-m` so `src.*` imports resolve.

```bash
uv sync                                          # install deps (incl. dev group: ruff)
uv run python -m src.ingestion.data_ingestion    # bronze -> silver (data/raw CSVs -> data/processed/*.parquet)
uv run python -m src.ingestion.build_features    # silver -> gold (data/gold/year=/month=/ parquet)
uv run python -m src.ml.train                    # rolling-origin CV of naive & seasonal-naive baselines (MAE)
uv run ruff check . && uv run ruff format .      # lint / format
```

`tests/` exists but contains no tests yet; there is no configured test runner.

`data/` is gitignored. Raw Kaggle files must be placed manually in `data/raw/` (`sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`).

## Architecture

Medallion pipeline; all paths, expected row counts, column lists and training config live in `src/config/config.py`.

- **Bronze → Silver** (`src/ingestion/data_ingestion.py`, pandas): calendar and prices are loaded fully and downcast (numerics → smallest unsigned/float, strings → `category`). The wide sales CSV is read in `CHUNK_SIZE` row chunks, each melted to long format, joined to calendar (`d` → date, events, per-state SNAP collapsed into a single `snap` column) and prices (`store_id, item_id, wm_yr_wk`), then written as `data/processed/part_NNNN.parquet`. Chunking exists because the melted result is ~58M rows (`SALES_MELTED_ROWS`).
- **Data quality** (`src/data_quality/checks.py`): bronze checks are pandas assertions on schema/row counts; silver checks are DuckDB SQL queries over the processed parquet glob (duplicates, date continuity, invalid sales/prices, null prices before/after launch). `validated_silver_sales` runs the silver suite at the end of ingestion. Checks fail via `assert`.
- **Silver → Gold** (`src/ingestion/build_features.py`, DuckDB): features are built as a chain of functions, each taking a DuckDB relation and returning a new projected relation (window functions partitioned by `item_id, store_id` ordered by `date`); some take `con` to join intermediate relations. Groups: product lifecycle (`launch_date` = first non-null price, `is_available`, `days_since_launch`), calendar, lags, sales intermittency, recent demand (rolling means), sales magnitude. Lags/rolling stats null out sales where the item wasn't available (pre-launch). Output is written partitioned by `year, month` with zstd. DuckDB is used instead of pandas because pandas OOM'd on the full dataset.
- **Modelling** (`src/ml/train.py`): loads only `item_id, store_id, date, sales` from gold via pyarrow (memory constraint), builds `N_FOLDS` rolling-origin folds of `FORECAST_HORIZON` (28) days ending at the last date, and scores baselines from each series' last 7 training days.

Note: `build_features.py` and `train.py` call their entry function at module level (no `__main__` guard), so importing them executes the full pipeline.

## Conventions

- Memory is the main constraint throughout — prefer DuckDB over the parquet files, column pruning, and downcasting over materialising full pandas frames.
- Design reasoning is recorded in `docs/`: `decision_logs.md` (pipeline decisions as Observation → Decision), `feature_decisions.md` (each feature tied to an EDA finding and whether it targets sales likelihood vs. magnitude), `analysis_logs.md` (EDA), `data_quality_checks.md` / `data_checks_logs.md`. Keep these in sync when changing pipeline or feature logic.
- Notebooks in `notebooks/` are exploratory; the `src/` modules are the source of truth.
