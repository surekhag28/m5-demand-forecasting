from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.config import (
    CUTOFF,
    FEATURES,
    MODEL_DIR,
    TARGET,
)
from src.ingestion.build_features_direct import (
    apply_encodings,
    create_encoding_features,
)
from src.ml.evaluate import evaluate, evaluate_test, get_params, save_artifacts
from src.utils.train_utils import load_data, make_folds, split

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(name):

    train_df = load_data(include_test=False, cutoff=CUTOFF)

    encs = create_encoding_features(train_df, train_df["date"].dt.date.max())
    train_df = apply_encodings(train_df, encs)

    train_data = lgb.Dataset(
        train_df[FEATURES],
        label=train_df[TARGET],
        params=get_params(),
        free_raw_data=True,
    )
    model = lgb.train(get_params(), train_data)

    save_artifacts(model, encs, name)


def run():

    start = time.time()
    df = load_data(include_test=False, cutoff=CUTOFF)

    print(f"Features loaded: {len(df)}, time taken:{time.time() - start}")

    start = time.time()

    start_date = df["date"].min().date()
    end_date = df["date"].max().date()

    folds = make_folds(start_date, end_date)

    metric = {
        "lightgbm": [],
    }

    for i, fold in enumerate(folds):
        print("train --> ", fold["train_start"], fold["train_end"])
        print("valid --> ", fold["valid_start"], fold["valid_end"])

        train_df, valid_df = split(df, fold)

        encs = create_encoding_features(train_df, fold["train_end"])
        train_df = apply_encodings(train_df, encs)
        valid_df = apply_encodings(valid_df, encs)

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]

        X_valid = valid_df[FEATURES]
        y_valid = valid_df[TARGET]

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

        model = lgb.train(get_params(), train_data, callbacks=[lgb.log_evaluation(50)])

        y_pred = pd.Series(model.predict(X_valid), index=y_valid.index)
        valid_df["prediction"] = y_pred

        # for wrmsse
        _, wrmsse = evaluate(train_df, valid_df)
        metric["lightgbm"].append(wrmsse)

    fold_cols = [f"fold_{i + 1}_wrmsse" for i in range(len(folds))]

    metric_df = pd.DataFrame.from_dict(
        metric,
        orient="index",
        columns=fold_cols,
    ).reset_index()

    metric_df["mean_wrmsse"] = metric_df[fold_cols].mean(axis=1)
    print(metric_df)

    print(
        f"Weighted root mean squared scaled error for lightgbm: {np.mean(metric['lightgbm'])}"
    )
    print(f"Total time take: {(time.time() - start) / 60} mins")


if __name__ == "__main__":
    model_name = "lgb_global_final"
    run()  # for cross validation
    train(model_name)
    evaluate_test(model_name)
