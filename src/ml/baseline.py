from __future__ import annotations

import pandas as pd

from src.config.config import CUTOFF, FORECAST_HORIZON, KEY, PROCESSED_DIR
from src.ml.evaluate import evaluate
from src.utils.utils import set_run_id


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


def make_naive_baseline():

    run_id = set_run_id("naive")
    train_df, valid_df = load_data()

    last_sales = train_df.loc[train_df["d"] == CUTOFF, KEY + ["sales"]].rename(
        columns={"sales": "prediction"}
    )

    valid_df = valid_df.merge(last_sales, on=KEY, how="left", validate="many_to_one")
    valid_df["prediction"] = valid_df["prediction"].fillna(
        0
    )  # new items release after cutoff

    level_scores, wrmsse = evaluate(train_df, valid_df)
    print(f"Naive baseline wrmsse for run_id {run_id}: {wrmsse}")


def make_seasonal_baseline():
    run_id = set_run_id("seasonal_naive")
    train_df, valid_df = load_data()

    train_df["wday"] = train_df["date"].dt.dayofweek
    valid_df["wday"] = valid_df["date"].dt.dayofweek
    last_week_sales = train_df[
        (train_df["d"] > CUTOFF - 7) & (train_df["d"] <= CUTOFF)
    ][KEY + ["sales", "wday"]].rename(columns={"sales": "prediction"})

    valid_df = valid_df.merge(last_week_sales, on=KEY + ["wday"], how="left")
    level_scores, wrmsse = evaluate(train_df, valid_df)
    print(f"Seasonl Naive baseline wrmsse for run_id {run_id}: {wrmsse}")


def run():
    make_naive_baseline()
    make_seasonal_baseline()


run()
