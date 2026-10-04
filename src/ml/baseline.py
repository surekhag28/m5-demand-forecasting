from __future__ import annotations

from src.config.config import CUTOFF, KEY
from src.ml.evaluate import evaluate
from src.utils.train_utils import load_data
from src.utils.utils import set_run_id


def make_naive_baseline():

    run_id = set_run_id("naive")
    train_df, valid_df = load_data()

    train_df = train_df[
        KEY + ["d", "sales", "date", "cat_id", "dept_id", "state_id", "sell_price"]
    ]
    valid_df = valid_df[KEY + ["sales", "date", "cat_id", "dept_id", "state_id", "d"]]

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

    train_df = train_df[
        KEY + ["d", "sales", "date", "cat_id", "dept_id", "state_id", "sell_price"]
    ]
    valid_df = valid_df[KEY + ["sales", "date", "cat_id", "dept_id", "state_id", "d"]]

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
