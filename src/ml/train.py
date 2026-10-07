from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.config import (
    CUTOFF,
    FEATURE_GROUPS,
    FEATURES,
    MODEL_DIR,
    TARGET,
)
from src.ml.evaluate import load_artifacts, save_artifacts
from src.utils.train_utils import get_params, load_data, make_folds, split

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_scale_and_weights(train_df_full):
    from src.ml.evaluate import calculate_hierarchy_scales, calculate_hierarchy_weights

    scale = calculate_hierarchy_scales(train_df_full)
    weights = calculate_hierarchy_weights(train_df_full)

    return scale, weights


def score_wrmsse(valid_df, scale_df, weights):
    from src.ml.evaluate import calculate_hierarchy_mse

    mse_df = calculate_hierarchy_mse(valid_df)

    metrics = scale_df.merge(mse_df, on=["level", "series_id"], how="inner").merge(
        weights, on=["level", "series_id"], how="inner"
    )

    metrics["rmsse"] = np.where(
        metrics["mse"] > 0, np.sqrt(metrics["mse"] / metrics["scale"]), 0.0
    )

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

    wrmsse = level_scores["wrmsse_level"].mean()

    return wrmsse


def train(model_name):
    params = get_params()

    df = load_data(include_test=False, cutoff=CUTOFF)
    df = df[df["year"].between(2015, 2016)]

    end_date = df["date"].max()
    valid_start = end_date - pd.Timedelta(days=27)

    train_df = df[df["date"] < valid_start]
    valid_df = df[df["date"] >= valid_start]

    del df
    print(train_df["date"].max(), valid_df["date"].min(), valid_df["date"].max())
    print(len(train_df))
    print(len(valid_df))

    train_data = lgb.Dataset(
        train_df[FEATURES],
        label=train_df[TARGET],
        params=params,
        free_raw_data=True,
    )
    valid_data = lgb.Dataset(
        valid_df[FEATURES],
        label=valid_df[TARGET],
        params=params,
        free_raw_data=True,
    )

    del train_df, valid_df

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100),
        ],
    )

    print(f"Models best iteration: {model.best_iteration}")
    save_artifacts(model, model_name)
    print("LightGBM model trained on entire dataset")


def evaluate_test(model_name, filter=None):
    from src.utils.train_utils import load_data

    model = load_artifacts(model_name)
    test_df = load_data(extra_filters=filter, include_test=True, cutoff=CUTOFF)
    train_df_full = load_data(include_test=False, cutoff=CUTOFF)

    test_df["prediction"] = model.predict(test_df[FEATURES])

    scale, weights = get_scale_and_weights(train_df_full)
    del train_df_full

    wrmsse = score_wrmsse(test_df, scale, weights)
    print(f"Final wrmsse of model {model_name} is: {wrmsse}")
    return wrmsse


def cv(df, params, folds, features, do_permutation=False):

    metric = []
    perm_results = []

    df_cut = df[(df["year"] >= 2015) & (df["year"] <= 2016)]
    for i, fold in enumerate(folds):
        print("train --> ", fold["train_start"], fold["train_end"])
        print("valid --> ", fold["valid_start"], fold["valid_end"])

        train_df, valid_df = split(df_cut, fold)
        train_df_full, _ = split(df, fold)

        print(train_df["date"].min(), train_df["date"].max())
        print(valid_df["date"].min(), valid_df["date"].max())
        print(train_df_full["date"].min(), train_df_full["date"].max())

        scale, weight = get_scale_and_weights(train_df_full)
        del train_df_full

        train_data = lgb.Dataset(
            train_df[features], label=train_df[TARGET], free_raw_data=True
        )

        model = lgb.train(params, train_data, callbacks=[lgb.log_evaluation(50)])
        del train_df, train_data

        valid_df["prediction"] = model.predict(valid_df[features])

        base_wrmsse = score_wrmsse(valid_df, scale, weight)

        print(base_wrmsse)

        metric.append(
            {"model": "lightgbm_global", "fold": i + 1, "wrmsse": base_wrmsse}
        )

        rng = np.random.default_rng(42)
        if do_permutation:
            for group, cols in FEATURE_GROUPS.items():
                X_shuffled = valid_df[features].copy()

                idx = rng.permutation(len(valid_df))
                for col in cols:
                    X_shuffled[col] = X_shuffled[col].values[idx]

                valid_df["prediction"] = model.predict(X_shuffled[features])

                del X_shuffled

                s_wrmsse = score_wrmsse(valid_df, scale, weight)

                perm_results.append(
                    {"fold": i + 1, "group": group, "delta": s_wrmsse - base_wrmsse}
                )
                print(f"Shuffled group {group}, delta: {s_wrmsse - base_wrmsse:+.4f}")

            del valid_df

            gc.collect()

    metric_df = pd.DataFrame(metric).pivot(
        index="model", columns="fold", values="wrmsse"
    )
    metric_df["mean_wrmsse"] = metric_df.mean(axis=1)
    print(metric_df)

    print(
        f"Weighted root mean squared scaled error for lightgbm: {metric_df['mean_wrmsse'].iloc[0]}"
    )

    if do_permutation:
        perm = pd.DataFrame(perm_results).pivot(
            index="group", columns="fold", values="delta"
        )
        perm["mean"] = perm.mean(axis=1)
        fold_cols = [col for col in perm.columns if col != "mean"]
        perm["folds_positive"] = (perm[fold_cols] > 0).sum(axis=1)
        print(perm.sort_values("mean", ascending=False).round(4))

        return metric_df, perm

    return metric_df


if __name__ == "__main__":
    model_name = "lgb_global_final"
    params = get_params()

    df = load_data(include_test=False, cutoff=CUTOFF)
    # df["noise"] = np.random.default_rng(42).random(len(df), dtype=np.float32) # removing noise as model is not using it
    print(f"Data points: {len(df)}, {df['d'].min()}, {df['d'].max()}")

    dead = (
        df.groupby(["item_id", "store_id"])["sales"].agg(["sum", "std"]).reset_index()
    )
    print("series with zero sales in train window:", (dead["sum"] == 0).sum())
    print("series with constant sales in train window:", (dead["std"] == 0).sum())

    print(dead[dead["sum"] == 0][["item_id", "store_id"]])

    folds = make_folds(df["date"].min().date(), df["date"].max().date())

    # Step 1: baseline + permutation table
    metrics = cv(
        df, params, folds, FEATURES, do_permutation=True
    )  # for cross validation for lightgbm baseline

    # Step 3: Retraining the model with final features
    # train(model_name)
    # evaluate_test(model_name)
