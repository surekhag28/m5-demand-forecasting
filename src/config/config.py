from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

SALES_PATH = RAW_DIR / "sales_train_evaluation.csv"
CALENDAR_PATH = RAW_DIR / "calendar.csv"
PRICES_PATH = RAW_DIR / "sell_prices.csv"

RESULTS_PATH = PROJECT_ROOT / "data" / "results"


SILVER_SALES_PATH = PROCESSED_DIR / "*.parquet"

MODEL_DIR = PROJECT_ROOT / "models"


CHUNK_SIZE = 1000
CALENDAR_ROWS = 1969
PRICES_ROWS = 6841121
SALES_ROWS = 30490
SALES_MELTED_ROWS = 30490 * 1941

ID_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]

CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]

PRICE_COLUMNS = [
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
]

FINAL_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "wm_yr_wk",
    "d",
    "sales",
    "sell_price",
    "weekday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
]

# ------- TRAINING CONFIG

START_YEAR = 2013
END_YEAR = 2016
N_FOLDS = 3
FORECAST_HORIZON = 28
CUTOFF = 1913  # last training data point
FINAL_TRAIN_CUTOFF = 1941

KEY = ["item_id", "store_id"]
TARGET = "sales"

CAT_COLS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
]

GROUP_ENC = {
    "item": "item_id",
    "store": "store_id",
    "item_store": "item_id, store_id",
    "store_dept": "store_id, dept_id",
    "store_cat": "store_id, cat_id",
}

FEATURES = [
    "item_id",
    "store_id",
    "state_id",
    "cat_id",
    "dept_id",
    "month",
    # "year",
    # "weekday",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "sell_price",
    "event_name_1",
    "event_type_1",
    "snap",
    "lag_28",
    "lag_29",
    "lag_30",
    "lag_31",
    "lag_32",
    "lag_33",
    "lag_34",
    "lag_35",
    "lag_56",
    "rolling_positive_rate_7",
    "rolling_positive_rate_28",
    "rolling_zero_rate_28",
    "days_since_last_sale",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_positive_mean_7",
    "rolling_positive_mean_28",
    "rolling_positive_median_28",
    "rolling_positive_std_28",
    "rolling_positive_max_28",
    "price_unique_count",
    "price_max",
    "price_min",
    "price_mean",
    "price_norm",
    "price_weekly_change",
    "price_change_m",
    "price_change_y",
    "store_dept_rolling_avg",
    "enc_item_mean",
    "enc_store_mean",
    "enc_item_store_mean",
    "enc_store_dept_mean",
    "enc_store_cat_mean",
    "enc_item_std",
    "enc_store_std",
    "enc_item_store_std",
    "enc_store_dept_std",
    "enc_store_cat_std",
]


STATES = ["CA", "TX", "WI"]
STORES = [
    "CA_1",
    "CA_2",
    "CA_3",
    "CA_4",
    "TX_1",
    "TX_2",
    "TX_3",
    "WI_1",
    "WI_2",
    "WI_3",
]


FEATURE_GROUPS = {
    "ids": ["item_id", "store_id", "state_id", "cat_id", "dept_id"],
    # calendar
    "weekly_pattern": ["day_of_week", "is_weekend"],
    "seasonality": ["month", "week_of_year", "day_of_month"],
    # "year": ["year"],
    "events": ["event_name_1", "event_type_1"],
    "snap": ["snap"],
    "price_level": [
        "sell_price",
        "price_max",
        "price_min",
        "price_mean",
        "price_norm",
        "price_unique_count",
    ],
    "price_change": [
        "price_weekly_change",
        "price_change_m",
        "price_change_y",
    ],
    "lags": [
        "lag_28",
        "lag_29",
        "lag_30",
        "lag_31",
        "lag_32",
        "lag_33",
        "lag_34",
        "lag_35",
        "lag_56",
    ],
    "rolling_mean": ["rolling_mean_7", "rolling_mean_28", "rolling_mean_56"],
    "intermittency": [
        "rolling_positive_rate_7",
        "rolling_positive_rate_28",
        "rolling_zero_rate_28",
        "days_since_last_sale",
    ],
    "positive_sales_stats": [
        "rolling_positive_mean_7",
        "rolling_positive_mean_28",
        "rolling_positive_median_28",
        "rolling_positive_std_28",
        "rolling_positive_max_28",
    ],
    "store_dept_rolling": ["store_dept_rolling_avg"],
    "enc_item": [
        "enc_item_mean",
        "enc_item_std",
        "enc_item_store_mean",
        "enc_item_store_std",
    ],
    "enc_store": [
        "enc_store_mean",
        "enc_store_std",
        "enc_store_dept_mean",
        "enc_store_dept_std",
        "enc_store_cat_mean",
        "enc_store_cat_std",
    ],
}
