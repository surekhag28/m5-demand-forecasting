from __future__ import annotations

import duckdb as db

from src.config.config import (
    END_YEAR,
    FORECAST_HORIZON,
    GOLD_DIR,
    SILVER_SALES_PATH,
    START_YEAR,
)


def _read_silver_data():
    con = db.connect()

    con.execute("SET memory_limit='10GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")

    sales = con.sql(f"""
        SELECT item_id, store_id, state_id, cat_id, dept_id,d,
               date,wm_yr_wk, min(date) over(partition by item_id,store_id order by date) as launch_date,year, month, weekday,
               sell_price, event_name_1, event_type_1, snap, sales
        FROM read_parquet('{SILVER_SALES_PATH}')
        WHERE sell_price IS NOT NULL
          AND NOT isnan(sell_price)
          AND year>={START_YEAR} and year<={END_YEAR}
    """)

    return con, sales


def create_calendar_features(rel):
    return rel.project("""
                *,
                week(date) as week_of_year,
                isodow(date) as day_of_week,
                day(date) as day_of_month,
                case when isodow(date)>=6 then 1 else 0 end as is_weekend
        """)


def create_lag_features(rel, lags=(FORECAST_HORIZON, *range(29, 36), 56)):

    window = "over (partition by item_id,store_id order by date)"

    lag_cols = ",\n".join(
        f"lag(sales, {lag}) {window}::INTEGER as lag_{lag}" for lag in lags
    )

    return rel.project(f"*, {lag_cols}")


def create_sales_intermittency_features(rel, horizon=FORECAST_HORIZON):

    partition = "partition by item_id, store_id order by date"

    def window(size):
        return f"over({partition} rows between {horizon + size - 1} preceding and {horizon} preceding)"

    def rate(condition, size, name):
        return f"round(avg(({condition})::DOUBLE) {window(size)}, 2) as {name}"

    cols = ",\n".join(
        [
            rate("sales > 0", 7, "rolling_positive_rate_7"),
            rate("sales > 0", 28, "rolling_positive_rate_28"),
            rate("sales = 0", 28, "rolling_zero_rate_28"),
            f"""datediff('day', max(case when sales > 0 then date end) over({partition} rows between unbounded preceding and {horizon} preceding), date) as days_since_last_sale""",
        ]
    )

    return rel.query(
        "intermittency_input", f"select *, {cols} from intermittency_input"
    )


def create_recent_demand_features(rel, horizon=FORECAST_HORIZON):

    partition = "partition by item_id,store_id order by date"

    def window(size):
        return f"over({partition} rows between {horizon + size - 1} preceding and {horizon} preceding)"

    def rolling_avg(size, name):
        return f"round(avg(sales) {window(size)},2) as {name}"

    cols = ",\n".join(
        [
            rolling_avg(7, "rolling_mean_7"),
            rolling_avg(7, "rolling_mean_28"),
            rolling_avg(7, "rolling_mean_56"),
        ]
    )

    return rel.query("recent_demand", f"select *, {cols} from recent_demand")


def create_sales_magnitude_features(rel, horizon=FORECAST_HORIZON):

    partition = "partition by item_id,store_id order by date"

    def window(size):
        return f"over({partition} rows between {horizon + size - 1} preceding and {horizon} preceding)"

    def sales_magnitude(func_name, size, name):
        return f"round({func_name}(case when sales>0 then sales end) {window(size)},2) as {name}"

    cols = ",\n".join(
        [
            sales_magnitude("avg", 7, "rolling_positive_mean_7"),
            sales_magnitude("avg", 28, "rolling_positive_mean_28"),
            sales_magnitude("median", 28, "rolling_positive_median_28"),
            sales_magnitude("stddev", 28, "rolling_positive_std_28"),
            sales_magnitude("max", 28, "rolling_positive_max_28"),
        ]
    )

    return rel.query("sales_magnitude", f"select *, {cols} from sales_magnitude")


def create_price_features(rel):

    partition = "partition by item_id,store_id"

    cols = ",\n".join(
        [
            f"count(distinct sell_price) over({partition}) as price_unique_count",
            f"max(sell_price) over({partition}) as price_max",
            f"min(sell_price) over({partition}) as price_min",
            f"round(avg(sell_price) over({partition}),2) as price_mean",
            f"round(sell_price/max(sell_price) over({partition}),2) as price_norm",
            f"round(sell_price/lag(sell_price,7) over({partition} order by date),2) as price_weekly_change",
            f"round(sell_price/avg(sell_price) over({partition}, month, year),2) as price_change_m",
            f"round(sell_price/avg(sell_price) over({partition}, year),2) as price_change_y",
        ]
    )

    return rel.query("price_momentum", f"select *, {cols} from price_momentum")


def create_store_dept_rolling(rel, horizon=FORECAST_HORIZON):

    daily = rel.query(
        "p",
        """select store_id,dept_id,date, sum(sales) as dept_sales from p group by store_id,dept_id,date""",
    )

    rolled = daily.query(
        "d",
        f"""select store_id, dept_id, date, round(avg(dept_sales)
        over(partition by store_id, dept_id order by date rows between {horizon + 27} preceding and {horizon} preceding),2) as store_dept_rolling_avg 
        from d""",
    )

    return rel.join(rolled, "store_id,dept_id,date", how="left")


def create_encoding_features(rel, horizon):

    from src.config.config import GROUP_ENC

    cols = []
    for name, keys in GROUP_ENC.items():
        window = f"over (partition by {keys} order by date range between unbounded preceding and interval {horizon} days preceding)"
        cols.append(f"avg(sales) {window} as enc_{name}_mean")
        cols.append(f"stddev(sales) {window} as enc_{name}_std")

    cols = ",\n".join(cols)
    return rel.query("encoding_features", f"select *, {cols} from encoding_features")


def create_trend_features(rel):
    return rel.query(
        "trend_features",
        """

                    select *,rolling_mean_28/nullif(enc_item_store_mean,0) as trend_28_vs_long,
                            rolling_mean_7/nullif(rolling_mean_28,0) as trend_7_vs_28,
                            rolling_mean_28/nullif(rolling_mean_56,0) as trend_28_vs_56,
                            rolling_mean_28/nullif(enc_store_dept_mean,0) as item_vs_dept,
                            rolling_mean_28/nullif(enc_store_cat_mean,0) as item_vs_cat,
                            enc_item_store_mean/nullif(enc_item_mean,0) as item_store_vs_global
                        from trend_features

                    """,
    )


def build_features_direct():
    con, sales = _read_silver_data()

    sales = create_calendar_features(sales)
    sales = create_lag_features(sales)
    sales = create_sales_intermittency_features(sales, horizon=FORECAST_HORIZON)
    sales = create_recent_demand_features(sales, horizon=FORECAST_HORIZON)
    sales = create_sales_magnitude_features(sales, horizon=FORECAST_HORIZON)
    sales = create_price_features(sales)
    sales = create_store_dept_rolling(sales)
    sales = create_encoding_features(sales, horizon=FORECAST_HORIZON)
    sales = create_trend_features(sales)

    # sales.filter("store_id=='WI_2' AND item_id=='FOODS_1_030'").order("date").show()

    sales.order("store_id, item_id, date")
    sales.write_parquet(
        str(GOLD_DIR) + "/features.parquet",
        overwrite=True,
        compression="zstd",
    )

    print(f"Candidate features created: {GOLD_DIR}")
    con.close()


if __name__ == "__main__":
    build_features_direct()
