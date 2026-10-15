from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.config import (
    CUTOFF,
    FEATURE_GROUPS,
    FEATURES,
    FINAL_TRAIN_CUTOFF,
    MODEL_DIR,
    TARGET,
)
from src.ml.evaluate import save_artifacts
from src.utils.train_utils import (
    get_params,
    load_data,
    make_folds,
    split,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

hier_cols = [
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


import psutil


def mem(tag):
    used = psutil.Process().memory_info().rss / 1e9
    free = psutil.virtual_memory().available / 1e9
    print(f"[{tag}] process uses {used:.1f} GB | free RAM {free:.1f} GB", flush=True)


def train_and_evaluate(model_name, n_trees=400):
    final_params = {**get_params(), "num_iterations": n_trees}

    filter = [("year", ">=", 2015), ("year", "<=", 2016)]
    df = load_data(extra_filters=filter, include_test=False, cutoff=FINAL_TRAIN_CUTOFF)
    gc.collect()

    num_feats = [
        c
        for c in FEATURES
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
    ]
    df[num_feats] = df[num_feats].astype("float32")
    mem("after float32")

    end_date = df["date"].max()  # d_1941
    valid_start = end_date - pd.Timedelta(days=27)  # d_1914
    train_mask = df["date"] < valid_start

    print("train ends:", df.loc[train_mask, "date"].max().date())
    print("test:", valid_start.date(), "→", end_date.date())

    scale, weights = get_scale_and_weights(df.loc[train_mask, hier_cols])
    gc.collect()

    valid_df = df.loc[~train_mask].copy()
    train_data = lgb.Dataset(
        df.loc[train_mask, FEATURES],
        label=df.loc[train_mask, TARGET],
        free_raw_data=True,
    )
    del df
    gc.collect()

    model = lgb.train(final_params, train_data)
    save_artifacts(model, model_name)
    del train_data
    gc.collect()

    valid_df["prediction"] = model.predict(valid_df[FEATURES])
    del model
    gc.collect()

    print(
        "test rows:", len(valid_df), "| days:", valid_df["date"].nunique()
    )  # 853,720 | 28

    valid_df[OOF_COLS + ["prediction"]].rename(
        columns={"prediction": "pred_global"}
    ).to_parquet("./artifacts/test_pred_global.parquet", index=False)
    score = score_wrmsse(valid_df, scale, weights)
    print(f"Global model test WRMSSE ({n_trees} trees): {score:.4f}")

    del valid_df
    gc.collect()
    return score


def cv(
    df,
    params,
    features,
    is_feature_validation=False,
    log_path=None,
    tree_grid=None,
    oof_dir="./artifacts",
):

    metric = []
    fold_scores = []
    tree_rows = []

    def train_and_score(train_df, valid_df, feats, scale, weight):

        model = lgb.train(
            params,
            lgb.Dataset(train_df[feats], label=train_df[TARGET], free_raw_data=True),
        )
        valid_df["prediction"] = model.predict(valid_df[feats])
        score = score_wrmsse(valid_df, scale, weight)

        del model, valid_df
        gc.collect()

        return score

    def train_and_score_grid(train_df, valid_df, feats, scale, weight, fold_no):
        grid_parms = {**params, "num_iterations": max(tree_grid)}
        model = lgb.train(
            grid_parms,
            lgb.Dataset(train_df[feats], label=train_df[TARGET], free_raw_data=True),
        )

        for n in tree_grid:
            valid_df[f"pred_{n}"] = model.predict(valid_df[feats], num_iteration=n)
            valid_df["prediction"] = valid_df[f"pred_{n}"]
            score = score_wrmsse(valid_df, scale, weight)
            tree_rows.append({"trees": n, "fold": fold_no, "wrmsse": score})
            print(f"fold {fold_no} | trees {n:5d} | {score:.4f}", flush=True)

        pred_cols = [f"pred_{n}" for n in tree_grid]
        valid_df[hier_cols + pred_cols].to_parquet(
            f"{oof_dir}/oof_global_fold{fold_no}.parquet", index=False
        )

        del model, train_df, valid_df
        gc.collect()
        return score  # score at the largest tree count = the baseline row, last one in the tree grid

    def save(row):
        metric.append(row)

        report = pd.DataFrame(metric).pivot(
            index="model", columns="fold", values="wrmsse"
        )
        fold_cols = report.columns.to_list()

        deltas = report[fold_cols] - report.loc["baseline", fold_cols]
        report["mean_wrmsse"] = report[fold_cols].mean(axis=1)
        report["mean_delta"] = deltas.mean(axis=1)
        report["folds_worse"] = (deltas > 0).sum(axis=1)

        report.sort_values("mean_delta", ascending=False).round(4).to_csv(log_path)

        print(
            f"fold {row['fold']} | {row['model']:25s} | {row['wrmsse']:.4f}", flush=True
        )

    df_recent = df[(df["year"] >= 2015) & (df["year"] <= 2016)]
    folds = make_folds(df_recent["date"].min().date(), df_recent["date"].max().date())
    del df

    for i, fold in enumerate(folds):
        print(
            f"fold {i + 1}: train {fold['train_start']} → {fold['train_end']}, "
            f"valid {fold['valid_start']} → {fold['valid_end']}"
        )

        train_df, valid_df = split(df_recent, fold)
        valid_df = valid_df.copy()

        # train_df_full, _ = split(df, fold)

        scale, weight = get_scale_and_weights(train_df[hier_cols])

        if tree_grid:
            base_wrmsse = train_and_score_grid(
                train_df, valid_df, features, scale, weight, fold_no=i + 1
            )
        else:
            base_wrmsse = train_and_score(train_df, valid_df, features, scale, weight)

        print(f"baseline: {base_wrmsse:.4f}")
        save({"model": "baseline", "fold": i + 1, "wrmsse": base_wrmsse})

        fold_scores.append(base_wrmsse)

        if is_feature_validation:
            for group, cols in FEATURE_GROUPS.items():
                features_new = [feature for feature in features if feature not in cols]
                if len(features_new) == len(features):
                    continue

                d_wrmsse = train_and_score(
                    train_df, valid_df, features_new, scale, weight
                )
                print(
                    f"{group:18s} {d_wrmsse:.4f}  (delta {d_wrmsse - base_wrmsse:+.4f})"
                )
                save({"model": f"{group}_dropped", "fold": i + 1, "wrmsse": d_wrmsse})

        del train_df, valid_df
        gc.collect()

    metric_df = pd.DataFrame(metric).pivot(
        index="model", columns="fold", values="wrmsse"
    )
    fold_cols = metric_df.columns.to_list()

    deltas = metric_df[fold_cols] - metric_df.loc["baseline", fold_cols]
    metric_df["mean_wrmsse"] = metric_df[fold_cols].mean(axis=1)
    metric_df["mean_delta"] = deltas.mean(axis=1)
    metric_df["folds_worse"] = (deltas > 0).sum(axis=1)

    if tree_grid:
        tree_report = pd.DataFrame(tree_rows).pivot(
            index="trees", columns="fold", values="wrmsse"
        )
        tree_report["mean_wrmsse"] = tree_report.mean(axis=1)
        print("\nTree count report:")
        print(tree_report.round(4))
        tree_report.round(4).to_csv(f"{oof_dir}/global_tree_report.csv")

        best_n = int(tree_report["mean_wrmsse"].idxmin())
        print(f"\nBest tree count (lowest mean WRMSSE): {best_n}")

        metric_df.attrs["tree_report"] = tree_report
        metric_df.attrs["num_iterations"] = best_n
    else:
        metric_df.attrs["num_iterations"] = params["num_iterations"]

    print(metric_df)

    return metric_df


if __name__ == "__main__":
    model_name = "lgb_global_final"
    params = get_params()

    df = load_data(include_test=False, cutoff=CUTOFF)

    # Step 1: baseline + Feature Drop Validation (to find best features)
    # metrics = cv(
    #     df,
    #     params,
    #     FEATURES,
    #     is_feature_validation=False,
    #     log_path="./artifacts/cv_results_global.csv",
    #     tree_grid=[100, 200, 300, 400, 500, 750, 1000, 1500, 2000],
    # )
    # print("Chosen num_iterations:", metrics.attrs["num_iterations"])

    # Step 2: Final trained model - with best features + best number of trees + best params (from step 1) + final train cutoff (no cv)
    train_and_evaluate(model_name, n_trees=400)
