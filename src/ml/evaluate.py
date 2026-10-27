from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.config import MODEL_DIR

HIERARCHY_LEVELS = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "category": ["cat_id"],
    "department": ["dept_id"],
    "state_category": ["state_id", "cat_id"],
    "state_department": ["state_id", "dept_id"],
    "store_category": ["store_id", "cat_id"],
    "store_department": ["store_id", "dept_id"],
    "item": ["item_id"],
    "state_item": ["item_id", "state_id"],
    "item_store": ["item_id", "store_id"],
}


def calculate_hierarchy_scales(train_df: pd.DataFrame):
    results = []

    for level, group_cols in HIERARCHY_LEVELS.items():
        if group_cols:
            daily_sales = (
                train_df.groupby(by=group_cols + ["date"], as_index=False)["sales"]
                .sum()
                .sort_values(by=group_cols + ["date"])
                .reset_index(drop=True)
            )

            daily_sales["cum_sales"] = daily_sales.groupby(group_cols)["sales"].cumsum()
            sales_started = daily_sales["cum_sales"].shift(1) > 0

            daily_sales["previous_sales"] = daily_sales.groupby(group_cols)[
                "sales"
            ].shift(1)

            daily_sales["diff_sq"] = (
                (daily_sales["sales"] - daily_sales["previous_sales"])
                .pow(2)
                .where(sales_started)
            )

            scale_df = (
                daily_sales.groupby(by=group_cols)["diff_sq"]
                .mean()
                .reset_index(name="scale")
            )
            scale_df["series_id"] = scale_df[group_cols].agg("_".join, axis=1)

        else:
            daily_sales = (
                train_df.groupby(by="date", as_index=False)["sales"]
                .sum()
                .sort_values(by="date")
                .reset_index(drop=True)
            )

            daily_sales["diff_sq"] = daily_sales["sales"].diff().pow(2)
            scale_df = pd.DataFrame(
                {"series_id": ["total"], "scale": daily_sales["diff_sq"].mean()}
            )

        scale_df["level"] = level

        results.append(scale_df[["level", "series_id", "scale"]])

    return pd.concat(results, ignore_index=True)


def calculate_hierarchy_mse(val_df: pd.DataFrame):

    results = []

    for level, group_cols in HIERARCHY_LEVELS.items():
        if group_cols:
            daily = val_df.groupby(by=group_cols + ["date"], as_index=False).agg(
                actual=("sales", sum), prediction=("prediction", sum)
            )

            daily["error_sq"] = (daily["actual"] - daily["prediction"]).pow(2)
            mse_df = (
                daily.groupby(by=group_cols)["error_sq"].mean().reset_index(name="mse")
            )
            mse_df["series_id"] = mse_df[group_cols].agg("_".join, axis=1)
        else:
            daily = val_df.groupby(by="date", as_index=False).agg(
                actual=("sales", sum), prediction=("prediction", sum)
            )
            daily["error_sq"] = (daily["actual"] - daily["prediction"]).pow(2)
            mse_df = pd.DataFrame(
                {"series_id": ["total"], "mse": daily["error_sq"].mean()}
            )

        mse_df["level"] = level
        results.append(mse_df[["level", "series_id", "mse"]])

    return pd.concat(results, ignore_index=True)


def calculate_hierarchy_weights(train_df: pd.DataFrame):
    results = []

    weight_df = train_df[
        train_df["date"] > train_df["date"].max() - pd.Timedelta(days=28)
    ]

    weight_df["dollar_sales"] = weight_df["sales"] * weight_df["sell_price"]

    results = []
    for level, group_col in HIERARCHY_LEVELS.items():
        if group_col:
            level_weight = (
                weight_df.groupby(by=group_col, as_index=False)["dollar_sales"]
                .sum()
                .reset_index()
            )
            level_weight["series_id"] = (
                level_weight[group_col].astype(str).agg("_".join, axis=1)
            )
        else:
            level_weight = pd.DataFrame(
                {
                    "series_id": ["total"],
                    "dollar_sales": weight_df["dollar_sales"].sum(),
                }
            )

        level_weight["level"] = level
        results.append(level_weight[["level", "series_id", "dollar_sales"]])

    return pd.concat(results, ignore_index=True)


def get_scale_and_weights(train_df):
    from src.ml.evaluate import calculate_hierarchy_scales, calculate_hierarchy_weights

    scale = calculate_hierarchy_scales(train_df)
    weights = calculate_hierarchy_weights(train_df)

    return scale, weights


def score_wrmsse(valid_df, scale_df, weights, debug=False, tag=None):

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


def save_artifacts(model, model_name):

    model.save_model(MODEL_DIR / f"{model_name}.txt")


def load_artifacts(model_name):
    import lightgbm as lgb

    model = lgb.Booster(model_file=str(MODEL_DIR / f"{model_name}.txt"))

    return model
