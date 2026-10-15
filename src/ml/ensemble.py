from __future__ import annotations

import gc

import numpy as np
import pandas as pd

from src.config.config import (
    CUTOFF,
    MODEL_DIR,
    STORES,
    TARGET,
)
from src.ml.evaluate import load_artifacts
from src.utils.train_utils import load_data

MODEL_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_FEATURES = FEATURES = [
    # "item_id",
    # "store_id",
    # "state_id",
    # "cat_id",
    # "dept_id",
    "month",
    # "year",
    # "weekday",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    # "sell_price",
    "event_name_1",
    "event_type_1",
    "snap",
    "lag_28",
    "lag_29",
    "lag_30",
    "lag_31",
    "lag_32",
    "lag_33",
    "lag_34",
    "lag_35",
    "lag_56",
    "rolling_positive_rate_7",
    "rolling_positive_rate_28",
    # "rolling_zero_rate_28",
    "days_since_last_sale",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_positive_mean_7",
    "rolling_positive_mean_28",
    "rolling_positive_median_28",
    "rolling_positive_std_28",
    "rolling_positive_max_28",
    "price_unique_count",
    "price_max",
    "price_min",
    "price_mean",
    "price_norm",
    "price_weekly_change",
    "price_change_m",
    "price_change_y",
    "enc_item_mean",
    "enc_store_mean",
    "enc_item_store_mean",
    "enc_store_dept_mean",
    "enc_store_cat_mean",
    "enc_item_std",
    "enc_store_std",
    "enc_item_store_std",
    "enc_store_dept_std",
    "enc_store_cat_std",
    "trend_28_vs_long",
    "trend_7_vs_28",
    "trend_28_vs_56",
    "item_vs_dept",
    "item_vs_cat",
]

STORE_FEATURES = [
    "month",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "event_name_1",
    "event_type_1",
    "snap",
    "lag_28",
    "lag_29",
    "lag_30",
    "lag_31",
    "lag_32",
    "lag_33",
    "lag_34",
    "lag_35",
    "lag_56",
    "rolling_positive_rate_7",
    "rolling_positive_rate_28",
    "days_since_last_sale",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_positive_mean_7",
    "rolling_positive_mean_28",
    "rolling_positive_median_28",
    "rolling_positive_std_28",
    "rolling_positive_max_28",
    "price_unique_count",
    "price_max",
    "price_min",
    "price_mean",
    "price_norm",
    "price_weekly_change",
    "price_change_m",
    "price_change_y",
    "enc_item_mean",
    "enc_item_store_mean",
    "enc_store_dept_mean",
    "enc_store_cat_mean",
    "enc_item_std",
    "enc_item_store_std",
    "enc_store_dept_std",
    "enc_store_cat_std",
    "trend_28_vs_long",
    "trend_7_vs_28",
    "trend_28_vs_56",
    "item_vs_dept",
    "item_vs_cat",
]

COLS = [
    "item_id",
    "store_id",
    "cat_id",
    "dept_id",
    "state_id",
    "date",
    TARGET,
    "sell_price",
]


def get_scale_and_weights(train_df_full):
    from src.ml.evaluate import calculate_hierarchy_scales, calculate_hierarchy_weights

    scale = calculate_hierarchy_scales(train_df_full)
    weights = calculate_hierarchy_weights(train_df_full)

    return scale, weights


def score_wrmsse(valid_df, scale_df, weights, debug=False, tag=None):
    from src.ml.evaluate import calculate_hierarchy_mse

    mse_df = calculate_hierarchy_mse(valid_df)

    metrics = scale_df.merge(mse_df, on=["level", "series_id"], how="inner").merge(
        weights, on=["level", "series_id"], how="inner"
    )

    # ---- debug: find series with scale 0 or missing, before dropping them ----
    if debug:
        bad_mask = metrics["scale"].isna() | (metrics["scale"] <= 0)
        level_total = metrics.groupby("level")["dollar_sales"].transform("sum")

        bad = metrics.loc[
            bad_mask, ["level", "series_id", "scale", "mse", "dollar_sales"]
        ].copy()
        bad["weight"] = metrics.loc[bad_mask, "dollar_sales"] / level_total[bad_mask]
        bad["causes_inf"] = bad["weight"] > 0
        bad = bad.sort_values(["causes_inf", "weight"], ascending=False)

        print(
            f"\n[{tag}] scale 0/missing: {len(bad)} series, "
            f"{bad['causes_inf'].sum()} with weight > 0 (these caused inf)"
        )
        print(bad["level"].value_counts().sort_index().to_string())
        print(bad[bad["causes_inf"]].to_string(index=False))

        if tag is not None:
            bad.to_csv(f"./artifacts/zero_scale_series_{tag}.csv", index=False)

    metrics = metrics[metrics["scale"] > 0].copy()

    metrics["rmsse"] = np.sqrt(metrics["mse"] / metrics["scale"])

    metrics["weight"] = metrics["dollar_sales"] / metrics.groupby(by="level")[
        "dollar_sales"
    ].transform("sum")

    metrics["weighted_rmsse"] = metrics["rmsse"] * metrics["weight"]
    level_scores = (
        metrics.groupby(by="level")["weighted_rmsse"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_rmsse": "wrmsse_level"})
    )

    return level_scores["wrmsse_level"].mean()


def evaluate():
    filter = [("year", ">=", 2015), ("year", "<=", 2016)]
    df = load_data(extra_filters=filter, include_test=True, cutoff=CUTOFF)
    end_date = df["date"].max()
    test_start = end_date - pd.Timedelta(days=27)

    test_df = df[df["date"] >= test_start]

    print(
        "train ends:", (test_start - pd.Timedelta(days=1)).date()
    )  # expect 2016-04-24
    print("test:", test_start.date(), "→", end_date.date())

    scale, weights = get_scale_and_weights(df.loc[df["date"] < end_date, COLS])
    print("test rows:", len(test_df), "| days:", test_df["date"].nunique())

    del df
    gc.collect()

    global_model = load_artifacts("lgb_global_final")
    test_df["pred_global"] = global_model.predict(test_df[GLOBAL_FEATURES])
    del global_model
    gc.collect()

    store_preds = []
    for store in STORES:
        store_test_df = test_df[test_df["store_id"] == store]
        model = load_artifacts(f"lightgbm_{store}")
        store_test_df["pred_store"] = model.predict(store_test_df[STORE_FEATURES])
        store_preds.append(store_test_df[COLS + ["pred_store"]])
        del model, store_test_df
        gc.collect()

    final_store_df = pd.concat(store_preds, ignore_index=True)

    n_before = len(test_df)
    test_df = test_df.merge(
        final_store_df[["item_id", "store_id", "date", "pred_store"]],
        on=["item_id", "store_id", "date"],
        how="inner",
        validate="one_to_one",
    )
    assert len(test_df) == n_before, "Rows lost or duplicated in the merge"
    print("rows after merge:", len(test_df))
    del final_store_df
    gc.collect()

    results = {}
    for name, pred in [
        ("global only", test_df["pred_global"]),
        ("stores only", test_df["pred_store"]),
        ("average", (test_df["pred_global"] + test_df["pred_store"]) / 2),
    ]:
        test_df["prediction"] = pred
        results[name] = score_wrmsse(test_df, scale, weights)
        print(f"{name:12s} test WRMSSE: {results[name]:.4f}")

    test_df[COLS + ["pred_global", "pred_store", "prediction"]].to_parquet(
        "./artifacts/test_predictions_ensemble.parquet", index=False
    )
    test_df[COLS + ["pred_global", "pred_store", "prediction"]].to_csv(
        "./artifacts/test_predictions_ensemble.csv", index=False
    )

    return results


if __name__ == "__main__":
    evaluate()
