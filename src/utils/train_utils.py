from datetime import timedelta

import pandas as pd

from src.config.config import (
    CUTOFF,
    FORECAST_HORIZON,
    KEY,
    N_FOLDS,
    PROCESSED_DIR,
    TRAIN_COLS,
    VAL_COLS,
)


def load_data():
    df = pd.read_parquet(
        PROCESSED_DIR  # filters=[("year", ">=", 2014), ("year", "<=", 2016)]
    )

    train_df = df[df["d"] <= CUTOFF][
        KEY + ["d", "sales", "date", "cat_id", "dept_id", "state_id", "sell_price"]
    ]

    valid_df = df[(df["d"] > CUTOFF) & (df["d"] <= CUTOFF + FORECAST_HORIZON)][
        KEY + ["sales", "date", "cat_id", "dept_id", "state_id", "d"]
    ]

    del df

    return train_df, valid_df


def make_folds(df: pd.DataFrame) -> dict:

    folds = []

    start_date = df["date"].min().date()
    end_date = df["date"].max().date()

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
    train_df = df[df["date"] <= pd.Timestamp(fold["train_end"])][TRAIN_COLS]
    valid_df = df[
        (df["date"] >= pd.Timestamp(fold["valid_start"]))
        & (df["date"] <= pd.Timestamp(fold["valid_end"]))
    ][VAL_COLS]

    return train_df, valid_df
