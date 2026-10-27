from __future__ import annotations

import gc

import lightgbm as lgb
import mlflow
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
from src.ml.evaluate import get_scale_and_weights, save_artifacts, score_wrmsse
from src.ml.tracking import configure_mlflow, log_cv_setup
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

        mlflow.log_param("training_num_iterations", max(tree_grid))

        model = lgb.train(
            grid_parms,
            lgb.Dataset(train_df[feats], label=train_df[TARGET], free_raw_data=True),
        )

        for n in tree_grid:
            valid_df[f"pred_{n}"] = model.predict(valid_df[feats], num_iteration=n)
            valid_df["prediction"] = valid_df[f"pred_{n}"]
            score = score_wrmsse(valid_df, scale, weight)
            tree_rows.append({"trees": n, "fold": fold_no, "wrmsse": score})

            mlflow.log_metric(f"wrmsse_trees_{n}_fold_{fold_no}", score)

            print(f"fold {fold_no} | trees {n:5d} | {score:.4f}", flush=True)

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

        print(
            f"fold {row['fold']} | {row['model']:25s} | {row['wrmsse']:.4f}", flush=True
        )

    df_recent = df[(df["year"] >= 2015) & (df["year"] <= 2016)]
    folds = make_folds(df_recent["date"].min().date(), df_recent["date"].max().date())
    del df

    log_cv_setup(df_recent, params, features, folds)

    for i, fold in enumerate(folds):
        print(
            f"fold {i + 1}: train {fold['train_start']} → {fold['train_end']}, "
            f"valid {fold['valid_start']} → {fold['valid_end']}"
        )

        train_df, valid_df = split(df_recent, fold)
        valid_df = valid_df.copy()

        scale, weight = get_scale_and_weights(train_df[hier_cols])

        if tree_grid:
            mlflow.log_param("tree_grid", ", ".join(map(str, tree_grid)))
            base_wrmsse = train_and_score_grid(
                train_df, valid_df, features, scale, weight, fold_no=i + 1
            )
        else:
            base_wrmsse = train_and_score(train_df, valid_df, features, scale, weight)
            mlflow.log_metric(f"wrmsse_fold_{i + 1}", float(base_wrmsse))

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
                delta = d_wrmsse - base_wrmsse
                print(f"{group:18s} {d_wrmsse:.4f}  (delta {delta:+.4f})")
                save({"model": f"{group}_dropped", "fold": i + 1, "wrmsse": d_wrmsse})

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

    # tree_rows.append({"trees": n, "fold": fold_no, "wrmsse": score})
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


if __name__ == "__main__":
    configure_mlflow()
    params = get_params()

    df = load_data(include_test=False, cutoff=CUTOFF)

    # Step 1: baseline + cv
    # with mlflow.start_run(run_name="baseline_cv_3fold"):
    #     metrics = cv(
    #         df,
    #         params,
    #         FEATURES,
    #         is_feature_validation=False,
    #         log_path="./artifacts/cv_results_global.csv",
    #         tree_grid=None,
    #     )
    # print(metrics)

    # Step 2: Feature Drop Validation (to find best features)
    with mlflow.start_run(run_name="feature_drop_validation_3fold"):
        mlflow.set_tag("stage", "feature_validation")
        metrics = cv(
            df,
            params,
            FEATURES,
            is_feature_validation=True,
            log_path="./artifacts/cv_results_global.csv",
            tree_grid=None,
        )
    print(metrics)

    # Step 3: Number of trees
    with mlflow.start_run(run_name="trees_validation_3fold"):   
    # metrics = cv(
    #     df,
    #     params,
    #     FEATURES,
    #     is_feature_validation=False,
    #     log_path="./artifacts/cv_results_global.csv",
    #     tree_grid=[100, 200, 300, 400, 500, 750, 1000, 1500, 2000],
    # )
    # print(metrics)

    # Step 2: Final trained model - with best features + best number of trees + best params (from step 1) + final train cutoff (no cv)
    # train_and_evaluate(GLOBAL_MODEL_NAME, n_trees=400)
