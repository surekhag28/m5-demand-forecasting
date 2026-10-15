from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from src.config.config import (
    CUTOFF,
    FINAL_TRAIN_CUTOFF,
    MODEL_DIR,
    STORES,
    TARGET,
)
from src.ml.evaluate import save_artifacts
from src.utils.train_utils import load_data, make_folds, split

MODEL_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
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

params = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_iterations": 2000,
    "num_leaves": 63,
    "min_data_in_leaf": 300,
    "max_bin": 63,
    # Light sampling + regularization: each store has ~10x less data
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 0.1,
    "force_col_wise": True,
    "num_threads": psutil.cpu_count(logical=False),
    # Reproducibility
    "seed": 42,
    "data_random_seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "verbosity": -1,
}

OOF_COLS = [
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


def train_and_evaluate(n_trees=200):
    final_params = {**params, "num_iterations": n_trees}
    df = load_data(include_test=False, cutoff=FINAL_TRAIN_CUTOFF)  # loading full data

    df_recent = df[(df["year"] >= 2015) & (df["year"] <= 2016)]
    df_recent = df_recent.sort_values(by=["item_id", "store_id", "date"])
    del df
    gc.collect()

    print(df_recent[["store_id", "item_id", "date"]].head(10))

    end_date = df_recent["date"].max()
    valid_start = end_date - pd.Timedelta(days=27)

    print("train ends: ", (valid_start - pd.Timedelta(days=1)))
    print("test : ", valid_start.date(), " -> ", end_date.date())

    train_all = df_recent[df_recent["date"] < valid_start]
    scale, weights = get_scale_and_weights(train_all)
    del train_all
    gc.collect()

    store_preds = []
    for store in STORES:
        model_name = f"lightgbm_{store}"
        print("Running for store :", store)

        store_recent = df_recent[df_recent["store_id"] == store]

        train_df = store_recent[store_recent["date"] < valid_start]
        valid_df = store_recent[store_recent["date"] >= valid_start].copy()

        train_data = lgb.Dataset(
            train_df[FEATURES],
            label=train_df[TARGET],
            free_raw_data=True,
        )
        valid_data = lgb.Dataset(
            valid_df[FEATURES],
            label=valid_df[TARGET],
            free_raw_data=True,
        )

        model = lgb.train(
            final_params,
            train_data,
            num_boost_round=n_trees,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.log_evaluation(period=25)],  # print every 25 trees
        )

        valid_df["prediction"] = model.predict(
            valid_df[FEATURES], num_iteration=n_trees
        )
        store_preds.append(valid_df[OOF_COLS + ["prediction"]])

        save_artifacts(model, model_name)

        del model, train_df, valid_df, train_data, valid_data, store_recent
        gc.collect()

    final_store_pred_df = pd.concat(store_preds, ignore_index=True)
    print("\ntest rows: ", len(final_store_pred_df))  # expect 853,720
    assert not final_store_pred_df.duplicated(["store_id", "item_id", "date"]).any(), (
        "Duplicate rows in test predictions"
    )

    wrmsse = score_wrmsse(final_store_pred_df, scale, weights)
    print(f"Final WRMSSE for combined store models: {wrmsse:.4f}")

    final_store_pred_df.rename(columns={"prediction": "pred_store"}).to_parquet(
        "./artifacts/test_pred_store.parquet", index=False
    )
    final_store_pred_df.rename(columns={"prediction": "pred_store"}).to_csv(
        "./artifacts/test_pred_store.csv", index=False
    )

    del final_store_pred_df
    gc.collect()


def run_cv():

    df = load_data(include_test=False, cutoff=CUTOFF)  # loading full data

    df_recent = df[(df["year"] >= 2015) & (df["year"] <= 2016)]
    df_recent = df_recent.sort_values(by=["item_id", "store_id", "date"])

    print(df_recent.head(10))
    folds = make_folds(df_recent["date"].min().date(), df_recent["date"].max().date())
    del df

    fold_preds = {i + 1: [] for i in range(len(folds))}
    # TREE_GRID = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]

    TREE_GRID = [50, 100, 150, 200, 250, 300, 400]

    for store in STORES:
        store_recent = df_recent[df_recent["store_id"] == store]

        print(store_recent[["item_id", "store_id", "date"]].head(5))

        print("Running for store :", store)
        for i, fold in enumerate(folds):
            # print("train --> ", fold["train_start"], fold["train_end"])
            # print("valid --> ", fold["valid_start"], fold["valid_end"])

            train_df, valid_df = split(store_recent, fold)

            # print(train_df[["item_id", "store_id", "date"]].head())
            # print("\n\n")
            # print(valid_df[["item_id", "store_id", "date"]].head())
            # print(train_df["date"].min(), train_df["date"].max())
            # print(valid_df["date"].min(), valid_df["date"].max())

            train_data = lgb.Dataset(
                train_df[FEATURES], label=train_df[TARGET], free_raw_data=True
            )
            valid_data = lgb.Dataset(
                valid_df[FEATURES],
                label=valid_df[TARGET],
                reference=train_data,
                free_raw_data=True,
            )
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.log_evaluation(period=25)],  # print every 25 trees
            )

            for n in TREE_GRID:
                valid_df[f"pred_{n}"] = model.predict(
                    valid_df[FEATURES], num_iteration=n
                )

            PRED_COLS = [f"pred_{n}" for n in TREE_GRID]
            fold_preds[i + 1].append(valid_df[OOF_COLS + PRED_COLS])

            del model, train_df, valid_df, train_data, valid_data
            gc.collect()

        del store_recent
        gc.collect()

    rows = []
    for i, fold in enumerate(folds, start=1):
        print("Running concatenation across the stores......")
        all_preds = pd.concat(
            fold_preds[i], ignore_index=True
        )  # stacking all the rows vertically

        train_df_full, _ = split(df_recent, fold)
        scale, weights = get_scale_and_weights(train_df_full)
        del train_df_full

        for n in TREE_GRID:
            all_preds["prediction"] = all_preds[f"pred_{n}"]
            score = score_wrmsse(all_preds, scale, weights, debug=True, tag=f"fold{i}")
            rows.append({"trees": n, "fold": i, "wrmsse": score})

        all_preds.to_csv(f"./artifacts/all_pred_stores_{i}.csv")
        all_preds.to_parquet(f"./artifacts/oof_store_fold{i}.parquet", index=False)

        n_zero = (scale["scale"] <= 0).sum() + scale["scale"].isna().sum()
        print(f"fold {i}: {n_zero} series with scale 0 or missing (dropped in scoring)")

    tree_report = pd.DataFrame(rows).pivot(
        index="trees", columns="fold", values="wrmsse"
    )
    tree_report["mean_wrmsse"] = tree_report.mean(axis=1)
    tree_report.round(4).to_csv("./artifacts/store_tree_report.csv")
    print(tree_report.round(4))


if __name__ == "__main__":
    # run_cv()  # for cross validation
    train_and_evaluate(
        n_trees=200
    )  # final train with best number of trees (i.e 200 from run_cv)
