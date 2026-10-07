from datetime import timedelta

import pandas as pd
import psutil

from src.config.config import (
    CAT_COLS,
    FORECAST_HORIZON,
    GOLD_DIR,
    N_FOLDS,
)
from src.utils.utils import get_optimised_data


def load_data(extra_filters=None, include_test=False, cutoff=None):

    if not include_test:
        filters = [("d", "<=", cutoff)]
    else:
        filters = [("d", ">", cutoff), ("d", "<=", cutoff + FORECAST_HORIZON)]

    if extra_filters:
        filters.extend(extra_filters)

    df = pd.read_parquet(
        str(GOLD_DIR) + "/features.parquet",
        filters=filters,
        read_dictionary=CAT_COLS,
    )
    df = get_optimised_data(df)

    return df


def make_folds(start_date, end_date) -> dict:

    folds = []

    for i in range(N_FOLDS, 0, -1):
        train_end = end_date - timedelta(days=i * FORECAST_HORIZON)
        valid_start = train_end + timedelta(days=1)
        valid_end = valid_start + timedelta(days=FORECAST_HORIZON - 1)

        folds.append(
            {
                "folds": N_FOLDS - i + 1,
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


def get_params():
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "learning_rate": 0.1,
        "num_iterations": 500,
        "num_leaves": 64,
        "min_child_samples": 2000,
        "max_bin": 63,
        "subsample": 0.5,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "force_col_wise": True,
        "n_jobs": psutil.cpu_count(logical=False),  # physical cores only
        "random_state": 42,
        "verbose": -1,
    }

    return params
