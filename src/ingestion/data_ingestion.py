from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SALES_PATH = RAW_DIR / "sales_train_validation.csv"
CALENDAR_PATH = RAW_DIR / "calendar.csv"
PRICES_PATH = RAW_DIR / "sell_prices.csv"

CHUNK_SIZE = 1000

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

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


def load_calendar() -> pd.DataFrame:

    calendar = pd.read_csv(CALENDAR_PATH, usecols=CALENDAR_COLUMNS)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def load_price() -> pd.DataFrame:

    prices = pd.read_csv(PRICES_PATH, usecols=PRICE_COLUMNS)
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int16")
    prices["sell_price"] = prices["sell_price"].astype("float32")

    return prices


def process_sales_chunk(
    sales_chunk: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
):
    daily_columns = [col for col in sales_chunk.columns if col.startswith("d_")]
    sales_long = sales_chunk.melt(
        id_vars=ID_COLUMNS, value_vars=daily_columns, var_name="d", value_name="sales"
    )

    sales_long = pd.merge(
        sales_long, calendar, on="d", how="left", validate="many_to_one"
    )
    sales_long["snap"] = sales_long["snap_CA"].where(
        sales_long["state_id"] == "CA",
        sales_long["snap_TX"].where(
            sales_long["state_id"] == "TX",
            sales_long["snap_WI"].where(sales_long["state_id"] == "WI"),
        ),
    )

    sales_long = pd.merge(
        sales_long,
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    sales_long.drop(columns=["d", "snap_CA", "snap_TX", "snap_WI"], inplace=True)

    final_colums = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "wm_yr_wk",
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

    sales_long = sales_long[final_colums]

    return sales_long


def run_ingestion():

    PROCESSED_DIR.mkdir(exist_ok=True)

    print("Loading calendar and price dataset")

    calendar = load_calendar()
    prices = load_price()

    total_rows = 0
    chunk_number = 0
    summary = {}

    for sales_chunk in pd.read_csv(SALES_PATH, chunksize=CHUNK_SIZE):
        print(f"Processing chunk {chunk_number}: {len(sales_chunk)} sales chunk data")

        processed_chunk = process_sales_chunk(
            sales_chunk=sales_chunk, calendar=calendar, prices=prices
        )

        OUTPUT_PATH = PROCESSED_DIR / f"part_{chunk_number:04d}.parquet"

        processed_chunk.to_parquet(OUTPUT_PATH, index=False)

        chunk_rows = len(processed_chunk)
        total_rows += chunk_rows

        print(f" {chunk_rows} processed rows")
        print(f" saved to {OUTPUT_PATH}")

        chunk_number += 1

        del sales_chunk
        del processed_chunk  # release memory

    summary = {
        "total_sales_data": total_rows,
        "no_of_parquet_files": chunk_number,
        "output_path": OUTPUT_PATH,
    }

    print("\n Ingestion complete")
    print(f"Total processed rows from sales data: {total_rows}")
    print(f"Number of parquet files: {chunk_number}")
    print(f"processed files saved at {PROCESSED_DIR}")

    return summary


if __name__ == "__main__":
    run_ingestion()
