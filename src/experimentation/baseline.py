from __future__ import annotations

import gc

import mlflow
import pandas as pd

from src.config.config import CUTOFF, KEY
from src.ml.evaluate import score_wrmsse
from src.utils.logging_utils import get_logger
from src.utils.mlflow_utils import configure_mlflow
from src.utils.train_utils import load_data

log = get_logger(__name__)

COLS = [
    "item_id",
    "store_id",
    "d",
    "sales",
    "date",
    "cat_id",
    "dept_id",
    "state_id",
    "sell_price",
]


def make_naive_baseline():

    with mlflow.start_run(run_name="naive_baseline") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"stage": "naive_baseline", "run_id": run_id})

        filter = [("year", ">=", 2015), ("year", "<=", 2016)]
        df = load_data(extra_filters=filter, include_test=True, cutoff=CUTOFF)

        end_date = df["date"].max()
        test_start_date = end_date - pd.Timedelta(days=27)

        train_df = df[df["date"] < test_start_date][COLS]
        valid_df = df[df["date"] >= test_start_date][COLS]

        del df
        gc.collect()

        mlflow.log_params(
            {
                "num_train_rows": len(train_df),
                "train_start_date": str(pd.to_datetime(train_df["date"].min())),
                "train_start_end": str(pd.to_datetime(train_df["date"].max())),
                "num_test_rows": len(valid_df),
                "num_days": valid_df["date"].nunique(),
                "test_start_date": str(pd.to_datetime(valid_df["date"].min())),
                "test_start_end": str(pd.to_datetime(valid_df["date"].max())),
            }
        )

        mlflow.log_dict({"features": list(COLS)}, "configuration/features.json")

        last_sales = train_df.loc[train_df["d"] == CUTOFF, KEY + ["sales"]].rename(
            columns={"sales": "prediction"}
        )

        valid_df = valid_df.merge(
            last_sales, on=KEY, how="left", validate="many_to_one"
        )
        valid_df["prediction"] = valid_df["prediction"].fillna(
            0
        )  # new items release after cutoff

        wrmsse = score_wrmsse(train_df, valid_df)

        mlflow.log_metric("wrmsse_naive", round(float(wrmsse), 4))
        log.info("Naive baseline wrmsse for run_id %d is %.4f", run_id, wrmsse)

        return {"rund_id": run_id, "wrmsse_naive": wrmsse}


def make_seasonal_baseline():

    with mlflow.start_run(run_name="seasonal_baseline") as run:
        run_id = run.info.run_id

        mlflow.set_tags({"stage": "seasonal_baseline", "run_id": run_id})

        filter = [("year", ">=", 2015), ("year", "<=", 2016)]
        df = load_data(extra_filters=filter, include_test=True, cutoff=CUTOFF)

        end_date = df["date"].max()
        test_start_date = end_date - pd.Timedelta(days=27)

        train_df = df[df["date"] < test_start_date][COLS]
        valid_df = df[df["date"] >= test_start_date][COLS]

        train_df = train_df[COLS]
        valid_df = valid_df[COLS]

        mlflow.log_params(
            {
                "num_train_rows": len(train_df),
                "train_start_date": str(pd.to_datetime(train_df["date"].min())),
                "train_start_end": str(pd.to_datetime(train_df["date"].max())),
                "num_test_rows": len(valid_df),
                "test_start_date": str(pd.to_datetime(valid_df["date"].min())),
                "test_start_end": str(pd.to_datetime(valid_df["date"].max())),
            }
        )

        mlflow.log_dict({"features": list(COLS)}, "configuration/features.json")

        train_df["wday"] = train_df["date"].dt.dayofweek
        valid_df["wday"] = valid_df["date"].dt.dayofweek
        last_week_sales = train_df[
            (train_df["d"] > CUTOFF - 7) & (train_df["d"] <= CUTOFF)
        ][KEY + ["sales", "wday"]].rename(columns={"sales": "prediction"})

        valid_df = valid_df.merge(last_week_sales, on=KEY + ["wday"], how="left")
        wrmsse = score_wrmsse(train_df, valid_df)

        mlflow.log_metric("wrmsse_seasonal", wrmsse)
        log.info("Seasonl Naive baseline wrmsse for run_id %d is %.4f", run_id, wrmsse)

        return {"run_id": run_id, "wrmsse_seasonal": wrmsse}


def run():
    configure_mlflow()
    make_naive_baseline()
    make_seasonal_baseline()


if __name__ == "__main__":
    run()
