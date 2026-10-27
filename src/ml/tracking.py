from __future__ import annotations

import hashlib
import json
import subprocess

import mlflow
import pandas as pd


def configure_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("m5_retail_demand_forecasting")


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def log_cv_setup(df, params, features, folds):

    metadata = {
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "min_date": str(pd.to_datetime(df["date"].min())),
        "max_date": str(pd.to_datetime(df["date"].max())),
        "unique_dates": int(df["date"].nunique()),
        "unique_items": int(df["item_id"].nunique()),
        "unique_stores": int(df["store_id"].nunique()),
    }

    metadata_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()

    mlflow.log_params(
        {
            "model_family": "LightGBM",
            "num_iterations": int(params["num_iterations"]),
            "feature_count": len(features),
            "cv_folds": len(folds),
            "cv_stratgey": "time_series",
            "dataset_rows": len(df),
            "dataset_fingerprint": metadata_hash,
            "git_commit": get_git_commit(),
        }
    )

    mlflow.set_tags({"stage": "baseline_cv", "dataset_fingerprint": metadata_hash})

    mlflow.log_dict(metadata, "dataset/dataset_metadata.json")
    mlflow.log_dict({"features": list(features)}, "configuration/features.json")
    mlflow.log_dict(params, "configuration/lightgbm_params.json")

    mlflow.log_dict(
        {"folds": [{k: str(v) for k, v in fold.items()} for fold in folds]},
        "configuration/cv_folds.json",
    )
