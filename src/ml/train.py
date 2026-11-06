from __future__ import annotations

import gc
import tempfile

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow.models import infer_signature

from src.config.config import (
    FEATURES,
    FINAL_TRAIN_CUTOFF,
    GLOBAL_MODEL_NAME,
    MODEL_DIR,
    TARGET,
)
from src.ml.evaluate import get_scale_and_weights, score_wrmsse
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_cv_setup,
    log_feature_importance,
)
from src.utils.train_utils import (
    get_params,
    load_data,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger(__name__)

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


def run(model_name, n_trees=400):

    log.info("Training the lightgbm model with %d number of trees", n_trees)

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

    log.info("Train data ends: %s", df.loc[train_mask, "date"].max().date())
    log.info(
        "Test data: starts at %s and ends with %s", valid_start.date(), end_date.date()
    )

    scale, weights = get_scale_and_weights(df.loc[train_mask, hier_cols])

    gc.collect()

    with mlflow.start_run(run_name="final_lightgbm_global") as run:
        log_cv_setup(df, final_params, FEATURES)
        mlflow.set_tags(
            {
                "stage": "final_train_lightgbm",
                "model": GLOBAL_MODEL_NAME,
                "model_type": "lightgbm",
            }
        )

        valid_df = df.loc[~train_mask].copy()
        train_data = lgb.Dataset(
            df.loc[train_mask, FEATURES],
            label=df.loc[train_mask, TARGET],
            free_raw_data=True,
        )
        del df
        gc.collect()

        model = lgb.train(final_params, train_data)
        del train_data
        gc.collect()
        log_feature_importance(model)

        valid_df["prediction"] = model.predict(valid_df[FEATURES])

        log.info(
            "Test data rows : %d | total days: %d",
            len(valid_df),
            valid_df["date"].nunique(),
        )

        score = score_wrmsse(valid_df, scale, weights)

        log.info("Global model test WRMSSE score with %d trees: %.4f", n_trees, score)

        mlflow.log_metric("wrmsse_score", round(float(score), 4))
        mlflow.log_dict({"features": list(FEATURES)}, "configuration/features.json")
        mlflow.log_dict(final_params, "configuration/lightgbm_params.json")
        mlflow.log_params(
            {
                "num_test_rows": len(valid_df),
                "forecast_window": valid_df["date"].nunique(),
                "test_start_date": str(pd.to_datetime(valid_df["date"].min())),
                "test_end_date": str(pd.to_datetime(valid_df["date"].max())),
            }
        )

        # logging the model
        with tempfile.TemporaryDirectory() as tmpdir:
            X_sample = valid_df[FEATURES].head(10)
            mlflow.lightgbm.save_model(
                model,
                tmpdir + "/model",
                signature=infer_signature(X_sample, model.predict(X_sample)),
                input_example=X_sample.head(3),
            )

            mlflow.log_artifacts(
                tmpdir + "/model", artifact_path="model"
            )  # from tmpdir to mlflow artifacts/model/ path

        del valid_df, model
        gc.collect()

        return {"wrmsse_score": score, "run_id": run.info.run_id}


if __name__ == "__main__":
    setup_logging()
    configure_mlflow()

    # Final trained model - with best features + best number of trees + best params (from step 1) + final train cutoff (no cv)
    run(GLOBAL_MODEL_NAME, n_trees=400)
