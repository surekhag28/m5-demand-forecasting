from __future__ import annotations

import gc

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.config.config import get_settings
from src.ml.evaluate import score_wrmsse
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.mlflow_utils import configure_mlflow, log_cv_setup
from src.utils.train_utils import (
    load_data,
    make_folds,
    split,
)

setup_logging()
log = get_logger(__name__)
settings = get_settings()

TARGET = settings.data.target

FEATURES = [
    feature
    for group in settings.features.experiment.candidate_groups
    for feature in settings.features.groups[group]
    if feature not in settings.features.experiment.excluded_features
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


def cv(
    df,
    is_feature_validation=False,
    log_path=None,
    tree_grid=None,
    oof_dir="./artifacts",
):

    metric = []
    fold_scores = []
    tree_rows = []

    params = settings.training.lgbm_params

    def train_and_score(
        train_df,
        valid_df,
    ):

        model = lgb.train(
            params,
            lgb.Dataset(train_df[FEATURES], label=train_df[TARGET], free_raw_data=True),
        )
        valid_df["prediction"] = model.predict(valid_df[FEATURES])
        score = score_wrmsse(train_df, valid_df)

        del model, valid_df
        gc.collect()

        return score

    def train_and_score_grid(train_df, valid_df, fold_no):
        grid_parms = {**params, "num_iterations": max(tree_grid)}

        mlflow.log_param("training_num_iterations", max(tree_grid))

        model = lgb.train(
            grid_parms,
            lgb.Dataset(train_df[FEATURES], label=train_df[TARGET], free_raw_data=True),
        )

        for n in tree_grid:
            valid_df[f"pred_{n}"] = model.predict(valid_df[FEATURES], num_iteration=n)
            valid_df["prediction"] = valid_df[f"pred_{n}"]
            score = score_wrmsse(train_df, valid_df)
            tree_rows.append({"trees": n, "fold": fold_no, "wrmsse": score})

            mlflow.log_metric(f"wrmsse_trees_{n}_fold_{fold_no}", round(score, 4))

            log.info("fold: %d | trees: %d | score: %.4f ", fold_no, n, score)

        # saving predictions for every tree
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

        if log_path:
            report.sort_values("mean_delta", ascending=False).round(4).to_csv(log_path)

        log.info(
            "fold: %d |  %-25s |  %.4f",
            row["fold"],
            row["model"],
            row["wrmsse"],
        )

    df_recent = df.copy()
    folds = make_folds(df_recent["date"].min().date(), df_recent["date"].max().date())
    del df

    log_cv_setup(df_recent, params, FEATURES, folds)

    for i, fold in enumerate(folds):
        with mlflow.start_run(run_name=f"cv_fold_{i + 1}", nested=True):
            log.info(
                "fold %d: train: %s -> %s, valid: %s -> %s",
                (i + 1),
                fold["train_start"],
                fold["train_end"],
                fold["valid_start"],
                fold["valid_end"],
            )

            train_df, valid_df = split(df_recent, fold)
            valid_df = valid_df.copy()

            if tree_grid:
                mlflow.log_param("tree_grid", ", ".join(map(str, tree_grid)))
                base_wrmsse = train_and_score_grid(train_df, valid_df, fold_no=i + 1)
            else:
                base_wrmsse = train_and_score(train_df, valid_df)
                mlflow.log_metric(f"wrmsse_fold_{i + 1}", float(base_wrmsse))

            log.info("baseline: %.4f", base_wrmsse)
            save({"model": "baseline", "fold": i + 1, "wrmsse": base_wrmsse})

            fold_scores.append(base_wrmsse)

            if is_feature_validation:
                for group, cols in settings.features.groups.items():
                    features_new = [
                        feature for feature in FEATURES if feature not in cols
                    ]
                    if len(features_new) == len(FEATURES):
                        continue

                    d_wrmsse = train_and_score(train_df, valid_df, features_new)
                    delta = d_wrmsse - base_wrmsse
                    log.info("%-18s %.4f (delta %+.4f)", group, d_wrmsse, delta)
                    save(
                        {"model": f"{group}_dropped", "fold": i + 1, "wrmsse": d_wrmsse}
                    )

                    mlflow.log_metric(
                        f"drop_{group}_wrmsse_fold_{i + 1}", round(float(d_wrmsse), 4)
                    )
                    mlflow.log_metric(
                        f"drop_{group}_delta_fold_{i + 1}",
                        round(float(delta), 4),
                    )
                    mlflow.log_dict(
                        {
                            "dropped_group": group,
                            "dropped_features": list(cols),
                            "remaining_features": list(features_new),
                            "feature_counts": len(features_new),
                        },
                        f"feature_validation/{group}_dropped/features.json",
                    )

            del train_df, valid_df
            gc.collect()

    scores = np.asarray(fold_scores, dtype=float)
    mlflow.log_metric("wrmsse_mean", round(float(scores.mean()), 4))
    mlflow.log_metric("wrmsse_std", round(float(scores.std()), 4))
    mlflow.log_metric("wrmsse_worst_fold", round(float(scores.max()), 4))

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
        log.info("\nTree count report:")
        log.info(tree_report.round(4))
        tree_report.round(4).to_csv(f"{oof_dir}/global_tree_report.csv")

        best_n = int(tree_report["mean_wrmsse"].idxmin())
        log.info("\nBest tree count (lowest mean WRMSSE): %d", best_n)

        metric_df.attrs["tree_report"] = tree_report
        metric_df.attrs["num_iterations"] = best_n

        tree_report_json = (
            tree_report.reset_index()
            .rename(
                columns={1: "fold_1_wrmsse", 2: "fold_2_wrmsse", 3: "fold_3_wrmsse"}
            )
            .round(4)
            .to_dict(orient="records")
        )

        mlflow.log_metric("lowest_mean_wrmsse", tree_report["mean_wrmsse"].min())
        mlflow.log_param("best_num_iterations", best_n)
        mlflow.log_dict(
            {"tree_validation": tree_report_json},
            "trees_validation/trees_wrmsse.json",
        )
    else:
        metric_df.attrs["num_iterations"] = params["num_iterations"]

    return metric_df


def run_baseline(df):
    # Step 1: baseline + cv
    with mlflow.start_run(run_name="baseline_cv_3fold"):
        metrics = cv(
            df,
            is_feature_validation=False,
            log_path="./artifacts/cv_results_global.csv",
            tree_grid=None,
        )
    log.info(metrics)


def run_feature_validation(df):
    # Step 2: Feature Drop Validation (to find best features)
    with mlflow.start_run(run_name="feature_drop_validation_3fold"):
        mlflow.set_tag("stage", "feature_validation")
        metrics = cv(
            df,
            is_feature_validation=True,
            log_path="./artifacts/cv_results_global.csv",
            tree_grid=None,
        )
        log.info(metrics)


def run_tree_validation(df):
    # Step 3: Number of trees
    with mlflow.start_run(run_name="trees_validation_3fold"):
        metrics = cv(
            df,
            is_feature_validation=False,
            log_path="./artifacts/cv_results_global.csv",
            tree_grid=settings.training.tree_grid,
        )
        log.info(metrics)


if __name__ == "__main__":
    configure_mlflow()

    filter = [("year", ">=", 2015), ("year", "<=", 2016)]
    df = load_data(
        include_test=False, extra_filters=filter, cutoff=settings.data.training_cutoff
    )

    run_baseline(df)

    # run_feature_validation(df, params)

    # run_tree_validation(df, params)
