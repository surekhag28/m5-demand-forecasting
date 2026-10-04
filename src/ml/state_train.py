from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config.config import (
    CUTOFF,
    FEATURES,
    MODEL_DIR,
    STATES,
    TARGET,
)
from src.ingestion.build_features_direct import (
    apply_encodings,
    create_encoding_features,
)
from src.ml.evaluate import evaluate, evaluate_test, get_params, save_artifacts
from src.utils.train_utils import load_data, make_folds, split

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_final(model_name, filter):
    from src.utils.utils import set_run_id

    run_id = set_run_id(model_name)

    train_df = load_data(
        extra_filters=filter,
        include_test=False,
        cutoff=CUTOFF,
    )

    encs = create_encoding_features(train_df, train_df["date"].dt.date.max())
    train_df = apply_encodings(train_df, encs)

    train_data = lgb.Dataset(
        train_df[FEATURES],
        label=train_df[TARGET],
        params=get_params(),
        free_raw_data=True,
    )
    model = lgb.train(get_params(), train_data)

    save_artifacts(model, encs, model_name)

    print(f"Training completed for model {model_name} with run_id {run_id}")


def train(folds, df: pd.DataFrame):
    params = get_params()

    scores = []

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

        train_data = lgb.Dataset(
            X_train, label=y_train, params=params, free_raw_data=True
        )

        model = lgb.train(params, train_data, callbacks=[lgb.log_evaluation(50)])

        y_pred = pd.Series(model.predict(X_valid), index=y_valid.index)
        valid_df["prediction"] = y_pred

        # for wrmsse
        _, wrmsse = evaluate(train_df, valid_df)
        scores.append(wrmsse)
        print(f"WRMSSE of fold {i} is : {wrmsse}")

        del train_df
        del valid_df

    return scores


def run_state():

    metric = {}

    for state in STATES:
        start = time.time()
        filter = [("state_id", "=", state)]
        df = load_data(filter, include_test=False, cutoff=CUTOFF)
        start_date = df["date"].min().date()
        end_date = df["date"].max().date()

        print(f"Features loaded: {len(df)}, time taken:{time.time() - start}")

        metric["lightgbm_" + state] = []

        folds = make_folds(start_date, end_date)
        wrmsse_scores = train(folds, df)

        metric["lightgbm_" + state].extend(wrmsse_scores)

    fold_cols = [f"fold_{i + 1}_wrmsse" for i in range(len(folds))]

    metric_df = pd.DataFrame.from_dict(
        metric,
        orient="index",
        columns=fold_cols,
    ).reset_index()

    metric_df["mean_wrmsse"] = metric_df[fold_cols].mean(axis=1)
    print(f"For state {state}, metrics: \n")
    print(metric_df)


def run():

    scores = []
    for state in STATES:
        model_name = f"lgb_state_{state}_final"

        filter = [("state_id", "=", state)]
        train_final(model_name, filter)
        score = evaluate_test(model_name, filter)
        scores.append(score)

    print(
        f"Final WRMSSE score of state specific models: {np.round(np.mean(scores), 2)}"
    )


if __name__ == "__main__":
    # run_state() # for cross validation
    run()
