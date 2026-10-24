from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.config.config import (
    CUTOFF,
    FEATURE_GROUPS,
    FEATURES,
    MODEL_DIR,
    N_TRIALS,
    TARGET,
)
from src.ml.evaluate import load_artifacts, save_artifacts
from src.utils.train_utils import (
    get_hpo_params,
    get_params,
    load_data,
    make_folds,
    split,
)

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


def suggest_params(trial):
    fixed_params = get_hpo_params()

    return {
        **fixed_params,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 5000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "tweedie_variance_power": trial.suggest_float(
            "tweedie_variance_power", 1.05, 1.5
        ),
    }


def auto_hpo(df, folds, features):

    def objective(trial):
        params = suggest_params(trial)

        def on_fold_end(fold_idx, score):
            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metric_df = cv(
            df,
            params,
            folds,
            features,
            on_fold_end=on_fold_end,
            is_feature_validation=False,
            is_hpo=True,
        )
        trial.set_user_attr("num_iterations", metric_df.attrs["num_iterations"])
        return float(metric_df.loc["baseline", "mean_wrmsse"])

    study = optuna.create_study(
        study_name="lightgbm_global_tuning",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    # first trial = existing current params used for cv training, so tuning starts from here
    study.enqueue_trial(
        {
            # "num_iterations": 500,
            "num_leaves": 64,
            "min_data_in_leaf": 2000,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "lambda_l2": 1e-3,
            "tweedie_variance_power": 1.1,
        },
        skip_if_exists=True,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print(f"best mean WRMSSE: {round(study.best_value, 4)}")
    print(f"best params: {study.best_params}")

    print("\n\n Param Importances")
    print(optuna.importance.get_param_importances(study))
    print("\n\n Trials DataFrame")
    print(study.trials_dataframe())
    print(
        study.trials_dataframe()
        .sort_values("value")[
            ["number", "value", "state"]
            + [c for c in study.trials_dataframe().columns if c.startswith("params_")]
        ]
        .head(10)
    )


def cv(
    df,
    params,
    folds,
    features,
    on_fold_end=None,
    is_feature_validation=False,
    is_hpo=False,
    log_path=None,
):

    metric = []
    fold_scores = []
    best_iters = []

    def train_and_score(train_df, valid_df, feats, scale, weight, is_hpo=False):

        if is_hpo:
            cut = train_df["date"].max() - pd.Timedelta(days=27)
            fit_df = train_df[train_df["date"] < cut]
            stop_df = train_df[train_df["date"] >= cut]

            fit_data = lgb.Dataset(
                fit_df[feats], label=fit_df[TARGET], free_raw_data=False
            )
            stop_data = lgb.Dataset(
                stop_df[feats],
                label=stop_df[TARGET],
                reference=fit_data,
                free_raw_data=False,
            )

            hpo_params = {k: v for k, v in params.items() if k != "num_iterations"}
            seed_scores, seed_best_iters = [], []

            for seed in [1, 2, 3]:
                model = lgb.train(
                    params={**hpo_params, "seed": seed},
                    train_set=fit_data,
                    num_boost_round=3000,
                    valid_sets=[stop_data],
                    valid_names=["valid"],
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=50, first_metric_only=True, verbose=False
                        ),
                        lgb.log_evaluation(period=0),
                    ],
                )
                valid_df["prediction"] = model.predict(
                    valid_df[feats]
                )  # uses best iteration
                seed_scores.append(score_wrmsse(valid_df, scale, weight))
                seed_best_iters.append(model.best_iteration)

                del model
                gc.collect()

            del fit_df, stop_df, fit_data, stop_data
            gc.collect()
            return np.mean(seed_scores), int(np.mean(seed_best_iters))

        else:
            model = lgb.train(
                params,
                lgb.Dataset(
                    train_df[feats], label=train_df[TARGET], free_raw_data=True
                ),
            )
            valid_df["prediction"] = model.predict(valid_df[feats])
            del model
            gc.collect()
            return score_wrmsse(valid_df, scale, weight), params["num_iterations"]

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

    for i, fold in enumerate(folds):
        # print("train --> ", fold["train_start"], fold["train_end"])
        # print("valid --> ", fold["valid_start"], fold["valid_end"])

        train_df, valid_df = split(df_recent, fold)

        train_df_full, _ = split(df, fold)
        train_df_full = train_df_full[
            [
                "item_id",
                "store_id",
                "cat_id",
                "dept_id",
                "state_id",
                "date",
                TARGET,
                "sell_price",
            ]
        ]
        scale, weight = get_scale_and_weights(train_df_full)
        del train_df_full

        # print("train dates:", train_df["date"].min(), "→", train_df["date"].max())
        # print("valid dates:", valid_df["date"].min(), "→", valid_df["date"].max())

        # baseline
        base_wrmsse, best_iter = train_and_score(
            train_df, valid_df, features, scale, weight, is_hpo=is_hpo
        )
        print(f"baseline: {base_wrmsse:.4f}")
        save({"model": "baseline", "fold": i + 1, "wrmsse": base_wrmsse})

        fold_scores.append(base_wrmsse)
        best_iters.append(best_iter)

        if is_hpo and on_fold_end is not None:
            running_mean = sum(fold_scores) / len(fold_scores)
            on_fold_end(i, running_mean)

        if is_feature_validation:
            for group, cols in FEATURE_GROUPS.items():
                features_new = [feature for feature in features if feature not in cols]
                if len(features_new) == len(features):
                    continue

                d_wrmsse, _ = train_and_score(
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

    metric_df.attrs["num_iterations"] = int(np.mean(best_iters))
    print(metric_df)

    return metric_df


if __name__ == "__main__":
    model_name = "lgb_global_final"
    params = get_params()

    df = load_data(include_test=False, cutoff=CUTOFF)
    # print(f"Data points: {len(df)}, {df['d'].min()}, {df['d'].max()}")

    dead = (
        df.groupby(["item_id", "store_id"])["sales"].agg(["sum", "std"]).reset_index()
    )
    # print("series with zero sales in train window:", (dead["sum"] == 0).sum())
    # print("series with constant sales in train window:", (dead["std"] == 0).sum())

    # print(dead[dead["sum"] == 0][["item_id", "store_id"]])

    folds = make_folds(df["date"].min().date(), df["date"].max().date())

    # Step 1: baseline + Feature Drop Validation (to find best features)
    # metrics = cv(
    #     df,
    #     params,
    #     folds,
    #     FEATURES,
    #     is_feature_validation=True,
    #     log_path="./artifacts/cv_results.csv",
    # )

    # Step 2: HPO - to find best params (from step 1) with freezing the best features

    # auto_hpo(df, folds, FEATURES)

    # Step 3: Re-evaluate the tuned model with best features and params from step 1 and step 2 (cv)
    # metrics = cv(
    #     df,
    #     params,
    #     folds,
    #     FEATURES,
    #     is_feature_validation=True,
    #     log_path="./artifacts/cv_results.csv",
    # )

    # Step 4: Final trained model - with best features + best params (from step 1 and step 2) + final train cutoff (no cv)
