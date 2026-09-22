# M5 Demand Forecasting

End-to-end retail demand forecasting project using the Kaggle M5 Forecasting Accuracy dataset.

## Objective

Build an end-to-end demand forecasting solution for retail products using historical sales, calendar information, and product pricing.

The project progressively covers:

- [x] Data understanding and EDA
- [x] Data ingestion
- [x] Data wrangling and quality checks
- [x] Time series analysis
- [x] Feature engineering
- [x] Baseline forecasting
- [ ] Statistical forecasting
- [ ] Machine learning forecasting
- [ ] Model evaluation
- [ ] ML engineering
- [ ] Azure deployment
- [ ] Model monitoring and drift detection
- [ ] Alerting and retraining

## Dataset

The project uses the M5 Forecasting Accuracy dataset, which contains Walmart retail sales data across products, stores, categories, calendar information, and weekly prices.

Dataset link: [m5-forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

## Architecture

High level architecture at every layer in the platform - data ingestion, wrangling etc


![M5 Demand Forecasting Architecture](docs/images/architecture.png)

## Data Pipeline

The pipeline follows a bronze / silver / gold layering approach:

- **Bronze (raw)** — `data/raw/`: original `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` files from Kaggle, validated for expected schema and row counts (`src/data_quality/checks.py`).
- **Silver (processed)** — `data/processed/`: sales data is read from the wide-format CSV in chunks, melted into a long item-store-date format, enriched with calendar and price attributes, downcasted to reduce memory usage, and written out as partitioned Parquet files (`src/ingestion/data_ingestion.py`). Validated against duplicate keys, continuous date ranges, and invalid/missing sales or price values.
- **Gold (features)** — `data/gold/`: candidate model features (product lifecycle, calendar/seasonality, lag, intermittency, recent demand level, and sales magnitude features) are built on top of the silver layer using DuckDB and written out partitioned by year/month (`src/ingestion/build_features.py`).

The reasoning behind each pipeline decision (chunked processing, downcasting, using DuckDB over Pandas for feature engineering, etc.) is captured in [`docs/decision_logs.md`](docs/decision_logs.md).

## Data Quality

Data quality checks run at both the bronze and silver layers to catch issues before they propagate downstream — schema/row-count validation on raw files, and duplicate, continuity, null/negative value, and pre/post-launch price checks on the processed sales data (`src/data_quality/checks.py`).

See [`docs/data_quality_checks.md`](docs/data_quality_checks.md) and [`docs/data_checks_logs.md`](docs/data_checks_logs.md) for details.

## Feature Engineering

Candidate features are grouped into: sales likelihood/intermittency, sales demand magnitude, recent demand (lags), recent demand level (rolling means), product lifecycle, and calendar/seasonality. Each feature is tied back to an EDA observation and a business question (sales likelihood vs. sales magnitude).

See [`docs/feature_decisions.md`](docs/feature_decisions.md) for the full feature catalogue and reasoning, and [`docs/analysis_logs.md`](docs/analysis_logs.md) for the underlying EDA.

## Modelling

`src/ml/train.py` implements a rolling-origin cross-validation scheme (`N_FOLDS` folds of `FORECAST_HORIZON` days each, see `src/config/config.py`) and evaluates naive and seasonal-naive baselines using MAE. These baselines set the reference point for the statistical and ML forecasting models planned next.

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.13+.

```bash
# install dependencies
uv sync

# download the M5 dataset from Kaggle and place the CSVs in data/raw/:
#   data/raw/sales_train_validation.csv
#   data/raw/calendar.csv
#   data/raw/sell_prices.csv

# run the ingestion pipeline (bronze -> silver)
uv run python -m src.ingestion.data_ingestion

# build candidate features (silver -> gold)
uv run python -m src.ingestion.build_features

# run baseline model evaluation
uv run python -m src.ml.train
```

## Project Structure

```text
m5-demand-forecasting/
│
├── data/
│   ├── raw/            # bronze: original Kaggle CSVs
│   ├── processed/      # silver: cleaned, long-format sales parquet
│   └── gold/           # gold: model-ready features, partitioned by year/month
│
├── docs/
│   ├── images/architecture.png
│   ├── analysis_logs.md         # EDA findings
│   ├── data_quality_checks.md   # data quality check definitions
│   ├── data_checks_logs.md      # data quality/analysis run logs
│   ├── decision_logs.md         # pipeline design decisions
│   └── feature_decisions.md     # candidate feature catalogue and reasoning
│
├── notebooks/
│   ├── 01_basic_eda.ipynb
│   ├── 02_eda_sql.ipynb
│   ├── analysis.ipynb
│   └── ingestion.ipynb
│
├── src/
│   ├── config/          # paths, schema, and training config
│   ├── data_quality/    # bronze/silver data quality checks
│   ├── ingestion/       # ingestion (bronze->silver) and feature building (silver->gold)
│   ├── ml/              # cross-validation and baseline model training
│   └── utils/           # shared helpers
│
├── tests/
│
├── main.py
├── README.md
├── pyproject.toml
└── uv.lock
```
