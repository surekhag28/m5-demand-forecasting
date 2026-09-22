from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error

from src.config.config import FORECAST_HORIZON, GOLD_DIR, N_FOLDS

KEY = ["item_id", "store_id"]


def load_dataset() -> pd.DataFrame:

    table = pq.read_table(
        source=GOLD_DIR,
        columns=["item_id", "store_id", "date", "sales"],
        read_dictionary=["item_id", "store_id"],
    )
    df = table.to_pandas(date_as_object=False)
    return df.sort_values(by="date", kind="stable", ignore_index=True)


def create_cv_folds(df: pd.DataFrame) -> dict:

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


def create_naive_baseline(last_week: pd.DataFrame, valid_df: pd.DataFrame):

    last_sales = last_week.groupby(by=KEY).tail(1)

    last_sales = last_sales[KEY + ["sales"]].rename(
        columns={"sales": "predicted_sales"}
    )

    merged = valid_df.merge(last_sales, on=KEY, how="left")
    return mean_absolute_error(merged["actual_sales"], merged["predicted_sales"])


def create_seasonal_baseline(last_week: pd.DataFrame, valid_df: pd.DataFrame):

    last_sales = last_week.copy()
    last_sales["position"] = last_sales.groupby(by=KEY, observed=True).cumcount()
    seasonal = last_sales[KEY + ["position", "sales"]].rename(
        columns={"sales": "predicted_sales"}
    )

    valid = valid_df.copy()
    valid["position"] = valid_df.groupby(by=KEY, observed=True).cumcount() % 7
    merged = valid.merge(seasonal, on=KEY + ["position"], how="left")
    return mean_absolute_error(merged["actual_sales"], merged["predicted_sales"])


def run():

    start = time.time()
    df = load_dataset()
    folds = create_cv_folds(df)

    mae_metric = {"naive_baseline": [], "seasonal_baseline": []}

    for i, fold in enumerate(folds):
        print("train --> ", fold["train_start"], fold["train_end"])
        print("valid --> ", fold["valid_start"], fold["valid_end"])

        train_df = df[df["date"] <= pd.Timestamp(fold["train_end"])]
        valid_df = df[
            (df["date"] >= pd.Timestamp(fold["valid_start"]))
            & (df["date"] <= pd.Timestamp(fold["valid_end"]))
        ].rename(columns={"sales": "actual_sales"})

        last_week = train_df.groupby(by=KEY, observed=True).tail(7)

        mae = create_naive_baseline(last_week, valid_df)
        mae_metric["naive_baseline"].append(mae)

        mae = create_seasonal_baseline(last_week, valid_df)
        mae_metric["seasonal_baseline"].append(mae)

    print(
        f"Mean abosulte error for naive baseline: {np.mean(mae_metric['naive_baseline'])}"
    )
    print(
        f"Mean abosulte error for seasonal baseline: {np.mean(mae_metric['seasonal_baseline'])}"
    )
    print(f"Total time take: {(time.time() - start) / 60} mins")


run()
