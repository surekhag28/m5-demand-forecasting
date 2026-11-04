from __future__ import annotations

import gc

import lightgbm as lgb
import mlflow
import pandas as pd

from src.config.config import get_settings
from src.ml.evaluate import score_wrmsse
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.mlflow_utils import configure_mlflow, log_cv_setup
from src.utils.train_utils import load_data, make_folds, split

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


def train(n_trees=200):
    final_params = {**settings.training.store_lgbm_params, "num_iterations": n_trees}

    filter = [("year", ">=", 2015), ("year", "<=", 2016)]
    df = load_data(
        include_test=True, extra_filters=filter, cutoff=settings.data.training_cutoff
    )  # loading full data

    end_date = df["date"].max()
    valid_start = end_date - pd.Timedelta(days=27)

    log.info("train ends: %s", (valid_start - pd.Timedelta(days=1)))
    log.info("test : %s -> %s", valid_start.date(), end_date.date())

    train_all = df[df["date"] < valid_start]

    store_preds = []
    for store in settings.data.stores:
        model_name = f"lightgbm_{store}"
        log.info("Running for store : %s", store)

        store_recent = train_all[train_all["store_id"] == store]

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

        del model, train_df, valid_df, train_data, valid_data, store_recent
        gc.collect()

    final_store_pred_df = pd.concat(store_preds, ignore_index=True)
    log.info("\ntest rows: %d", len(final_store_pred_df))  # expect 853,720
    assert not final_store_pred_df.duplicated(["store_id", "item_id", "date"]).any(), (
        "Duplicate rows in test predictions"
    )

    wrmsse = score_wrmsse(train_all, final_store_pred_df)
    log.info(f"Final WRMSSE for combined store models: {wrmsse:.4f}")

    final_store_pred_df.rename(columns={"prediction": "pred_store"}).to_parquet(
        "./artifacts/test_pred_store.parquet", index=False
    )
    final_store_pred_df.rename(columns={"prediction": "pred_store"}).to_csv(
        "./artifacts/test_pred_store.csv", index=False
    )

    del final_store_pred_df
    gc.collect()


def cv():

    filter = [("year", ">=", 2015), ("year", "<=", 2016)]
    df_recent = load_data(
        include_test=False, extra_filters=filter, cutoff=settings.data.training_cutoff
    )  # loading full data

    df_recent = df_recent.sort_values(by=["item_id", "store_id", "date"])

    folds = make_folds(df_recent["date"].min().date(), df_recent["date"].max().date())

    fold_preds = {i + 1: [] for i in range(len(folds))}

    log_cv_setup(df_recent, settings.training.store_lgbm_params, FEATURES, folds)

    TREE_GRID = [50, 100, 150, 200, 250, 300, 400]

    for store in settings.data.stores:
        store_recent = df_recent[df_recent["store_id"] == store]
        log.info("Running for store : %s", store)
        for i, fold in enumerate(folds):
            # log.info("train --> ", fold["train_start"], fold["train_end"])
            # log.info("valid --> ", fold["valid_start"], fold["valid_end"])

            train_df, valid_df = split(store_recent, fold)

            # log.info(train_df[["item_id", "store_id", "date"]].head())
            # log.info("\n\n")
            # log.info(valid_df[["item_id", "store_id", "date"]].head())
            # log.info(train_df["date"].min(), train_df["date"].max())
            # log.info(valid_df["date"].min(), valid_df["date"].max())

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
                settings.training.store_lgbm_params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                callbacks=[lgb.log_evaluation(period=25)],  # print every 25 trees
            )

            mlflow.log_params(
                {
                    "store_name": store,
                    "train_rows": len(train_df),
                    "train_start": train_df["date"].min(),
                    "train_end": train_df["date"].max(),
                    "valid_rows": len(valid_df),
                    "valid_start": valid_df["date"].min(),
                    "valid_end": valid_df["date"].max(),
                }
            )

            for n in TREE_GRID:
                valid_df[f"pred_{n}"] = model.predict(
                    valid_df[FEATURES], num_iteration=n
                )
                mlflow.log_param("tree_number", n)

            PRED_COLS = [f"pred_{n}" for n in TREE_GRID]
            fold_preds[i + 1].append(valid_df[OOF_COLS + PRED_COLS])

            del model, train_df, valid_df, train_data, valid_data
            gc.collect()

        del store_recent
        gc.collect()

    rows = []
    for i, fold in enumerate(folds, start=1):
        train_df, _ = split(df_recent, fold)

        log.info("Running concatenation across the stores......")
        all_preds = pd.concat(
            fold_preds[i], ignore_index=True
        )  # stacking all the rows vertically

        for n in TREE_GRID:
            all_preds["prediction"] = all_preds[f"pred_{n}"]
            score = score_wrmsse(train_df, all_preds, debug=True, tag=f"fold{i}")
            rows.append({"trees": n, "fold": i, "wrmsse": score})

        all_preds.to_csv(f"./artifacts/all_pred_stores_{i}.csv")
        all_preds.to_parquet(f"./artifacts/oof_store_fold{i}.parquet", index=False)

    tree_report = pd.DataFrame(rows).pivot(
        index="trees", columns="fold", values="wrmsse"
    )
    tree_report["mean_wrmsse"] = tree_report.mean(axis=1)
    tree_report.round(4).to_csv("./artifacts/store_tree_report.csv")
    log.info(tree_report.round(4))


if __name__ == "__main__":
    configure_mlflow()
    # cv()  # for cross validation
    train(n_trees=200)  # final train with best number of trees (i.e 200 from run_cv)
