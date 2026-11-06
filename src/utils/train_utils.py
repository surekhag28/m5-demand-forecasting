from datetime import timedelta
from pathlib import Path

import pandas as pd
import psutil

from src.config.config import get_settings
from src.utils.utils import get_optimised_data

settings = get_settings()
BUCKET = Path(settings.storage.bucket)
GOLD_BASE = BUCKET / settings.storage.gold


def load_data(extra_filters=None, include_test=False, cutoff=None):

    if not include_test:
        filters = [("d", "<=", cutoff)]
    else:
        filters = [("d", "<=", cutoff + settings.data.forecast_horizon)]

    if extra_filters:
        filters.extend(extra_filters)

    df = pd.read_parquet(
        GOLD_BASE,
        filters=filters,
        engine="pyarrow",
        read_dictionary=settings.features.cat_cols,
        partitioning="hive",
    )
    df = get_optimised_data(df)

    return df


def make_folds(start_date, end_date) -> dict:

    folds = []

    for i in range(settings.data.n_folds, 0, -1):
        train_end = end_date - timedelta(days=i * settings.data.forecast_horizon)
        valid_start = train_end + timedelta(days=1)
        valid_end = valid_start + timedelta(days=settings.data.forecast_horizon - 1)

        folds.append(
            {
                "folds": settings.data.n_folds - i + 1,
                "train_start": start_date,
                "train_end": train_end,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )

    return folds


def split(df: pd.DataFrame, fold: dict):
    train_df = df[df["date"] <= pd.Timestamp(fold["train_end"])]
    valid_df = df[
        (df["date"] >= pd.Timestamp(fold["valid_start"]))
        & (df["date"] <= pd.Timestamp(fold["valid_end"]))
    ]

    return train_df, valid_df


def get_hpo_params():
    return {
        "objective": "tweedie",
        "learning_rate": 0.1,  # fixed while tuning (fast); lower it for the final model
        "max_bin": 63,
        "bagging_freq": 1,
        "force_col_wise": True,
        "num_threads": psutil.cpu_count(logical=False),
        "seed": 42,
        "deterministic": True,
        "verbosity": -1,
    }
