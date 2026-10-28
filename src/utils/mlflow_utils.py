from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import hashlib
import json
import subprocess

import lightgbm as lgb
import matplotlib.pyplot as plt
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


def log_cv_setup(df, params, features, folds=None):

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
            "cv_folds": len(folds) if folds is not None else 0,
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

    if folds is not None:
        mlflow.log_dict(
            {"folds": [{k: str(v) for k, v in fold.items()} for fold in folds]},
            "configuration/cv_folds.json",
        )


def log_feature_importance(model, top_n=30):

    fi = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    )

    fi["gain_pct"] = fi["gain"] / fi["gain"].sum() * 100
    fi["split_pct"] = fi["split"] / fi["split"].sum() * 100

    fi = fi.sort_values(by="gain", ascending=False).reset_index(drop=True)

    mlflow.log_text(fi.to_csv(index=False), "feature_importance/feature_importance.csv")

    for imp_type in ["gain", "split"]:
        ax = lgb.plot_importance(
            model,
            importance_type=imp_type,
            max_num_features=top_n,
            figsize=(10, max(5, top_n * 0.3)),
            title=f"Top {top_n} features by {imp_type}",
        )

        mlflow.log_figure(
            ax.figure, f"feature_importance/top_{top_n}_features_{imp_type}.png"
        )
        plt.close(ax.figure)

    mlflow.log_metrics(
        {f"fi_gain_pct/{row.feature}": row.gain_pct for row in fi.head(10).itertuples()}
    )

    unused_features = fi[(fi["split"] == 0) | (fi["gain"] == 0)]["feature"].to_list()
    mlflow.log_param("num_unused_features", len(unused_features))
    mlflow.log_dict(
        {"unused_features": unused_features}, "feature_importance/unused_features.json"
    )
