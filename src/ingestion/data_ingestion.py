from __future__ import annotations

import pandas as pd

from src.config.config import (
    CALENDAR_COLUMNS,
    CALENDAR_PATH,
    CALENDAR_ROWS,
    CHUNK_SIZE,
    ID_COLUMNS,
    PRICE_COLUMNS,
    PRICES_PATH,
    PRICES_ROWS,
    PROCESSED_DIR,
    SALES_PATH,
)
from src.data_quality.checks import (
    validate_bronze_calendar,
    validate_bronze_price,
    validated_silver_sales,
)


def load_calendar() -> pd.DataFrame:
    """Loads calendar dataset from CSV file"""
    try:
        calendar = pd.read_csv(CALENDAR_PATH, usecols=CALENDAR_COLUMNS)
    except FileNotFoundError:
        print(f"Unable to read file: {CALENDAR_PATH}")
        raise

    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def load_price() -> pd.DataFrame:
    """Loads sell price dataset from CSV file"""
    try:
        prices = pd.read_csv(PRICES_PATH, usecols=PRICE_COLUMNS)
    except FileNotFoundError:
        print(f"Unable to read file: {PRICES_PATH}")
        raise

    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int16")
    prices["sell_price"] = prices["sell_price"].astype("float32")

    return prices


def process_sales_chunk(
    sales_chunk: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
):
    """Transforms historical sales data from wide to analysis ready format.

    Args:
        sales_chunk (pd.DataFrame): Chunk of historical sales data in wide format, containing item-store identifiers and daily sales columns.
        calendar (pd.DataFrame): Calendar data containing dates, weeks identifiers, events and state-level snap indicators.
        prices (pd.DataFrame): weekly sell-price data for item x store combinations.

    Returns:
        pd.DataFrame: Transformed sales data in long format, with one row per item-store-day observation and
                        enriched with calendar and price attributes.

    Returns:
        pd.Data
    """

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
    """Runs the sales data ingestion and transformation pipeline.

    Returns:
        dict[str,Any]: Summary of the ingestion run pipeline.
    """

    PROCESSED_DIR.mkdir(exist_ok=True)

    print("Loading calendar and price dataset")

    calendar = load_calendar()
    validate_bronze_calendar(calendar, CALENDAR_COLUMNS, CALENDAR_ROWS)

    prices = load_price()
    validate_bronze_price(prices, PRICE_COLUMNS, PRICES_ROWS)

    print("Bronze data quality checks completed")

    total_rows = 0
    chunk_number = 0
    summary = {}

    try:
        for sales_chunk in pd.read_csv(SALES_PATH, chunksize=CHUNK_SIZE):
            print(
                f"Processing chunk {chunk_number}: {len(sales_chunk)} sales chunk data"
            )

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
    except Exception as e:
        print(f"Failed to process file: {e}")
        raise

    validated_silver_sales()

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
