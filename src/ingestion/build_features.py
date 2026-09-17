from __future__ import annotations

import duckdb as db

from src.config.config import GOLD_SALES_PATH, SILVER_SALES_PATH


def _read_silver_data():
    con = db.connect()

    sales = con.from_query(f"""
            SELECT item_id,store_id,state_id,cat_id,dept_id,date,year,month,weekday,sell_price,event_name_1,event_type_1,snap,sales,
                FROM
                    read_parquet ('{SILVER_SALES_PATH}')
        """)

    return con, sales


def create_product_lifecycle_features(rel):

    return rel.project("""
            *,
            min(
                case when sell_price is not null then date
                end
            ) over(partition by item_id,store_id order by date) as launch_date,
            case
                when sell_price is not null then 1
                else 0
            end as is_available,
            case
                when date>=launch_date then datediff('day',launch_date,date)
                else null
            end as days_since_launch

        """)


def create_calendar_features(rel):
    return rel.project("""
                *,
                extract(
                    week
                    from
                        date
                ) as week_of_year,
                case
                    when weekday = 'Saturday'
                    or weekday = 'Sunday' then 1
                    else 0
                end as is_weekend
        """)


def create_lag_features(rel):

    return rel.project("""
                    *,
                    lag (
                        case
                            when is_available = 1 then sales
                            else null
                        end,
                        1
                    ) over (
                        partition by
                            item_id,
                            store_id
                        order by
                            date
                    ) as lag_1,
                    lag (
                        case
                            when is_available = 1 then sales
                            else null
                        end,
                        7
                    ) over (
                        partition by
                            item_id,
                            store_id
                        order by
                            date
                    ) as lag_7,
                    lag (
                        case
                            when is_available = 1 then sales
                            else null
                        end,
                        14
                    ) over (
                        partition by
                            item_id,
                            store_id
                        order by
                            date
                    ) as lag_14,
                    lag (
                        case
                            when is_available = 1 then sales
                            else null
                        end,
                        28
                    ) over (
                        partition by
                            item_id,
                            store_id
                        order by
                            date
                    ) as lag_28,
        """)


def create_sales_intermittency_features(con, rel):

    #     * `rolling_positive_rate_7`
    # * `rolling_positive_rate_28`
    # * `rolling_zero_rate_28`
    # * `days_since_last_sale`

    rel.create_view("intermittency_input", replace=True)

    return con.from_query("""
            with
                post_launch_sales as (
                    select
                        store_id,
                        item_id,
                        sales,
                        date
                    from
                        intermittency_input
                    where
                        is_available = 1
                ),
                intermittent_sales as (
                    select
                        store_id,
                        item_id,
                        date,
                        case
                            when count(*) over (
                                partition by
                                    store_id,
                                    item_id
                                order by
                                    date rows between 7 preceding
                                    and 1 preceding
                            ) = 7 then round(
                                avg(
                                    case
                                        when sales > 0 then 1.0
                                        else 0.0
                                    end
                                ) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 7 preceding
                                        and 1 preceding
                                ),
                                2
                            )
                            else null
                        end as rolling_positive_rate_7,
                        case
                            when count(*) over (
                                partition by
                                    store_id,
                                    item_id
                                order by
                                    date rows between 28 preceding
                                    and 1 preceding
                            ) = 28 then round(
                                avg(
                                    case
                                        when sales > 0 then 1.0
                                        else 0.0
                                    end
                                ) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 28 preceding
                                        and 1 preceding
                                ),
                                2
                            )
                            else null
                        end as rolling_positive_rate_28,
                        case
                            when count(*) over (
                                partition by
                                    store_id,
                                    item_id
                                order by
                                    date rows between 28 preceding
                                    and 1 preceding
                            ) = 28 then round(
                                avg(
                                    case
                                        when sales = 0 then 1.0
                                        else 0.0
                                    end
                                ) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 28 preceding
                                        and 1 preceding
                                ),
                                2
                            )
                            else null
                        end as rolling_zero_rate_28,
                        datediff (
                            'day',
                            max(
                                case
                                    when sales > 0 then date
                                end
                            ) over (
                                partition by
                                    store_id,
                                    item_id
                                order by
                                    date rows between unbounded preceding
                                    and 1 preceding
                            ),
                            date
                        ) as days_since_last_sale
                    from
                        post_launch_sales
                )
            select
                r.*,
                i.rolling_positive_rate_7,
                i.rolling_positive_rate_28,
                i.rolling_zero_rate_28,
                i.days_since_last_sale
            from
                intermittency_input as r
                left join intermittent_sales as i on r.store_id = i.store_id
                and r.item_id = i.item_id
                and r.date = i.date
            order by
                item_id,
                store_id,
                date
        """)


def create_recent_demand_features(con, rel):

    #     * `rolling_mean_7`
    # * `rolling_mean_28`
    # * `rolling_mean_56`

    rel.create_view("recent_sales_demand", replace=True)

    return con.from_query("""
                    with
                        post_launch as (
                            select
                                store_id,
                                item_id,
                                date,
                                sales
                            from
                                recent_sales_demand
                            where
                                is_available = 1
                        ),
                        recent_demand as (
                            select
                                store_id,
                                item_id,
                                date,
                                sales,
                                case
                                    when count(*) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 7 preceding
                                            and 1 preceding
                                    ) = 7 then round(avg(sales) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 7 preceding
                                            and 1 preceding
                                    ),2)
                                    else null
                                end as rolling_mean_7,
                                case
                                    when count(*) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 28 preceding
                                            and 1 preceding
                                    ) = 28 then round(avg(sales) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 28 preceding
                                            and 1 preceding
                                    ),2)
                                    else null
                                end as rolling_mean_28,
                                case
                                    when count(*) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 56 preceding
                                            and 1 preceding
                                    ) = 56 then round(avg(sales) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 56 preceding
                                            and 1 preceding
                                    ),2)
                                    else null
                                end as rolling_mean_56
                            from
                                post_launch
                        )
                    select
                        r.*,
                        d.rolling_mean_7,
                        d.rolling_mean_28,
                        d.rolling_mean_56
                    from
                        recent_sales_demand as r
                        left join recent_demand as d on r.store_id = d.store_id
                        and r.item_id = d.item_id
                        and r.date = d.date
                    order by
                        r.item_id,
                        r.store_id,
                        r.date
            """)


def create_sales_magnitude_features(con, rel):

    # * `rolling_positive_mean_7`
    # * `rolling_positive_mean_28`
    # * `rolling_positive_median_28`
    # * `rolling_positive_max_28`

    rel.create_view("sales_demand_magnitude", replace=True)

    return con.from_query("""
                with
                    post_launch as (
                        select
                            store_id,
                            item_id,
                            date,
                            sales
                        from
                            sales_demand_magnitude
                        where
                            is_available = 1
                    ),
                    sales_demand as (
                        select
                            store_id,
                            item_id,
                            date,
                            sales,
                            case
                                when count(*) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 7 preceding
                                        and 1 preceding
                                ) = 7 then round(
                                    avg(
                                        case
                                            when sales > 0 then sales
                                        end
                                    ) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 7 preceding
                                            and 1 preceding
                                    ),
                                    2
                                )
                                else null
                            end as rolling_positive_mean_7,
                            case
                                when count(*) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 28 preceding
                                        and 1 preceding
                                ) = 28 then round(
                                    avg(
                                        case
                                            when sales > 0 then sales
                                        end
                                    ) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 28 preceding
                                            and 1 preceding
                                    ),
                                    2
                                )
                                else null
                            end as rolling_positive_mean_28,
                            case
                                when count(*) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 28 preceding
                                        and 1 preceding
                                ) = 28 then round(
                                    median (
                                        case
                                            when sales > 0 then sales
                                        end
                                    ) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 28 preceding
                                            and 1 preceding
                                    ),
                                    2
                                )
                                else null
                            end as rolling_positive_median_28,
                            case
                                when count(*) over (
                                    partition by
                                        store_id,
                                        item_id
                                    order by
                                        date rows between 28 preceding
                                        and 1 preceding
                                ) = 28 then round(
                                    max(
                                        case
                                            when sales > 0 then sales
                                        end
                                    ) over (
                                        partition by
                                            store_id,
                                            item_id
                                        order by
                                            date rows between 28 preceding
                                            and 1 preceding
                                    ),
                                    2
                                )
                                else null
                            end as rolling_positive_max_28,
                        from
                            post_launch
                    )
                select
                    s.*,
                    d.rolling_positive_mean_7,
                    rolling_positive_mean_28,
                    rolling_positive_median_28,
                    rolling_positive_max_28
                from
                    sales_demand_magnitude as s
                    left join sales_demand as d on s.store_id = d.store_id
                    and s.item_id = d.item_id
                    and s.date = d.date
        """)


def build_features():
    con, sales = _read_silver_data()
    sales = create_product_lifecycle_features(sales)
    sales = create_calendar_features(sales)
    sales = create_lag_features(sales)
    sales = create_sales_intermittency_features(con, sales)
    sales = create_recent_demand_features(con, sales)
    sales = create_sales_magnitude_features(con, sales)

    # sales.show(max_rows=1000)
    sales.write_parquet(str(GOLD_SALES_PATH), overwrite=True)
    print(f"Candidate features created: {GOLD_SALES_PATH}")


build_features()
