from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedRowCounts(StrictConfig):
    calendar: PositiveInt
    prices: PositiveInt
    sales_wide: PositiveInt
    sales_melted: PositiveInt


class TrainingWindow(StrictConfig):
    start_year: PositiveInt = Field(
        description="First year included in the training window"
    )
    end_year: PositiveInt = Field(
        description="Last year included in the training window"
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> Self:
        if self.start_year >= self.end_year:
            raise ValueError("start_year must be less than end_year")

        return self


class DataSettings(StrictConfig):
    chunk_size: PositiveInt = Field(
        default=1000,
        gt=0,
        description="Number of records processed per ingestion chunk",
    )
    expected_row_counts: ExpectedRowCounts
    target: str = Literal["sales"]
    series_keys: list[Literal["item_id", "store_id"]] = Field(
        description="Required columns identifying item-store series"
    )
    forecast_horizon: PositiveInt = Field(
        description="Number of future days to forecast"
    )
    training_cutoff: PositiveInt = Field(
        description="Last training index in the dataset"
    )
    n_folds: PositiveInt
    training_window: TrainingWindow
    stores: list[str]

    @field_validator("series_keys")
    @classmethod
    def validate_keys(
        cls, columns: list[Literal["item_id", "store_id"]]
    ) -> list[Literal["item_id", "store_id"]]:
        if set(columns) != {"item_id", "store_id"}:
            raise ValueError(
                "series_keys must contain item_id and store_id to identify series"
            )

        return columns


class IngestionSettings(StrictConfig):
    id_columns: list[str] = Field(min_length=1)
    calendar_columns: list[str] = Field(min_length=1)
    price_columns: list[str] = Field(min_length=1)
    final_columns: list[str] = Field(min_length=1)

    @field_validator("id_columns", "calendar_columns", "price_columns", "final_columns")
    @classmethod
    def validate_columns(cls, columns: list[str]) -> list[str]:
        if any(not col.strip() for col in columns):
            raise ValueError("Column names cannot be empty")

        if len(columns) != len(set(columns)):
            raise ValueError("Column names need to be unique")

        return columns

    @model_validator(mode="after")
    def validate_final_schema(self) -> Self:
        required_cols = {
            "item_id",
            "store_id",
            "date",
            "d",
            "sales",
            "sell_price",
            "snap",
        }

        missing = required_cols - set(self.final_columns)

        if missing:
            raise ValueError(f"final_columns is missing {sorted(missing)} columns")

        return self


class FeatureExperimentSettings(StrictConfig):
    candidate_groups: list[str] = Field(
        min_length=1, description="Candidate features used for training the model"
    )
    excluded_features: list[str] = Field(
        description="Features excluded from training list"
    )


class FeatureTrainingSettings(StrictConfig):
    global_model_name: str
    recipe_path: Path


class FeatureSettings(StrictConfig):
    cat_cols: list[str]
    encodings: dict[str, list[str]]
    groups: dict[str, list[str]]
    experiment: FeatureExperimentSettings
    training: FeatureTrainingSettings


class StorageSettings(StrictConfig):
    bucket: str = Field(description="Storage bucket or container name")
    bronze: str = Field(description="Location to store raw data")
    silver: str = Field(description="Location to store processed data")
    gold: str = Field(description="Location to store features data for model training")

    sales_file: str
    cander_file: str
    price_file: str


class TrainingSettings(StrictConfig):
    lgbm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.1,
            "num_iterations": 500,
            "num_leaves": 64,
            "min_data_in_leaf": 2000,
            "max_bin": 63,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "feature_fraction": 1.0,
            "force_col_wise": True,
            "num_threads": 4,
            "seed": 42,
            "data_random_seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "verbosity": -1,
        }
    )

    store_lgbm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.05,
            "num_iterations": 2000,
            "num_leaves": 63,
            "min_data_in_leaf": 300,
            "max_bin": 63,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "lambda_l2": 0.1,
            "force_col_wise": True,
            "num_threads": 4,
            "seed": 42,
            "data_random_seed": 42,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "verbosity": -1,
        }
    )

    tree_grid: list[int] = Field(
        default_factory=lambda: [100, 200, 300, 400, 500, 750, 1000, 1500, 2000]
    )


class Settings(StrictConfig):
    storage: StorageSettings
    data: DataSettings
    ingestion: IngestionSettings
    features: FeatureSettings
    training: TrainingSettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    if not SETTINGS_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found: {SETTINGS_PATH}")

    with SETTINGS_PATH.open(mode="r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return Settings.model_validate(raw_config)


settings = get_settings()
print(settings.ingestion.id_columns)
