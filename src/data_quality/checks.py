from __future__ import annotations

import duckdb
import pandas as pd

from src.config.config import PROCESSED_DIR


def validate_bronze_calendar(
    calendar: pd.DataFrame, calendar_columns: list, expected_rows: int
) -> None:

    assert not calendar.empty, "Bronze DQ failed: calendar dataset in empty"

    missing_cols = set(calendar_columns) - set(calendar.columns)

    assert not missing_cols, (
        f"Bronze DQ failed: calendar missing columns: {missing_cols}"
    )

    assert len(calendar) == expected_rows, (
        f"Bronze DQ failed: expected rows: {expected_rows}, but got {len(calendar)} rows"
    )


def validate_bronze_price(
    prices: pd.DataFrame, prices_columns: list, expected_rows: int
) -> None:

    assert not prices.empty, "Bronze DQ failed: prices dataset is empty"

    missing_cols = set(prices_columns) - set(prices.columns)

    assert not missing_cols, f"Bronze DQ failed: prices missing columns: {missing_cols}"

    assert len(prices) == expected_rows, (
        f"Bronze DQ failed: expected rows: {expected_rows}, but got {len(prices)} rows"
    )


def validate_bronze_sales(sales_chunk: pd.DataFrame, sales_columns: list) -> None:

    assert not sales_chunk.empty, "Bronze DQ failed: sales chunk is empty"


def validate_silver_no_duplicates() -> None:
    # no duplicate (item, store, date) rows
    query = """
                SELECT
                    item_id,store_id,date,count(*) as row_count
                    from read_parquet(?)
                    group by item_id,store_id,date
                    having count(*)>1
            """

    duplicate = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert duplicate is None, (
        f"Silver DQ failed: duplicate key found at item - store - date level: {duplicate}"
    )


def validate_silver_continuous_dates() -> None:
    query = """
    
                select item_id,store_id,min(date),max(date),count(*) as actual_days,
                        DATE_DIFF('day', MIN(date), MAX(date)) + 1 as expected_days
                from read_parquet(?)
                group by item_id,store_id
                having count(*) != DATE_DIFF('day', MIN(date), MAX(date)) + 1
            """

    missing_dates = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert missing_dates is None, (
        f"Silver DQ failed: series with missing range found: {missing_dates}"
    )


def validate_silver_invalid_sales() -> None:

    query = """
        
            select item_id,store_id,date,sales
            from read_parquet(?)
            where sales < 0
            limit 1
        """

    invalid_sales = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert invalid_sales is None, (
        f"Silver DQ failed: negative sales found in series: {invalid_sales}"
    )


def validate_silver_no_sales() -> None:
    query = """
                    
                        select item_id,store_id,date,sales
                        from read_parquet(?)
                        where sales is null
                        limit 1
                    """

    null_sales = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert null_sales is None, (
        f"Silver DQ failed: null sales found in the series: {null_sales}"
    )


def validate_silver_invalid_price() -> None:

    query = """
        
            select item_id,store_id,date,sell_price
            from read_parquet(?)
            where sell_price <= 0
            limit 1
        """

    invalid_price = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert invalid_price is None, (
        f"Silver DQ failed: negative sell prices found in series: {invalid_price}"
    )


def validate_silver_null_price_before_launch() -> None:

    parquet_path = f"{PROCESSED_DIR}/*.parquet"

    query = """
            with launch as (
                select item_id,store_id,min(wm_yr_wk) as launch_week
                from read_parquet(?)
                where sell_price is not null
                group by item_id,store_id
            )

            select s.item_id,s.store_id,s.date,s.wm_yr_wk,s.sell_price
            from read_parquet(?) as s
            join launch as l
            on l.item_id=s.item_id
                and l.store_id=s.store_id
            where s.wm_yr_wk < l.launch_week
            and s.sell_price is null
            limit 1
    """

    expected_null_price = (
        duckdb.connect().execute(query, [parquet_path, parquet_path]).fetchone()
    )

    assert expected_null_price is not None, (
        f"Silver DQ failed: no expected pre-launch sell prices found :{expected_null_price}"
    )


def validate_silver_null_price_after_launch() -> None:
    query = """
            with launch as (
                select item_id,store_id,min(wm_yr_wk) as launch_week
                from read_parquet(?)
                where sell_price is not null
                group by item_id,store_id
            )

            select s.item_id,s.store_id,s.date,s.wm_yr_wk,s.sell_price
            from read_parquet(?) as s
            join launch as l
            on l.item_id=s.item_id
                and l.store_id=s.store_id
            where s.wm_yr_wk >= l.launch_week
            and s.sell_price is null
            limit 1
    """

    invalid_price = (
        duckdb.connect().execute(query, [f"{PROCESSED_DIR}/*.parquet"]).fetchone()
    )

    assert invalid_price is None, (
        f"Silver DQ failed: null sell prices after launch date for :{invalid_price}"
    )


def validated_silver_sales() -> None:
    validate_silver_no_duplicates()
    validate_silver_continuous_dates()
    validate_silver_invalid_sales()
    validate_silver_no_sales()
    validate_silver_invalid_price()
    validate_silver_null_price_before_launch()
    validate_silver_null_price_before_launch()


# if __name__ == "__main__":
#     validate_silver_invalid_price()
