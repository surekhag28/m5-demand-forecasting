from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SALES_PATH = RAW_DIR / "sales_train_validation.csv"
CALENDAR_PATH = RAW_DIR / "calendar.csv"
PRICES_PATH = RAW_DIR / "sell_prices.csv"


CHUNK_SIZE = 1000
CALENDAR_ROWS = 1969
PRICES_ROWS = 6841121
SALES_ROWS = 30490

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
