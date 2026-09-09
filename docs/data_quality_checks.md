## Data Quality Checks

## 1. Bronze Layer (raw data)

- Checks whether calendar, prices and sales dataset loads successfully
- checks for the presence of required columns in datasets
- checks for total records

## 2. Silver Layer (processed data)

Data quality checks are implemented on the Silver layer using DuckDB SQL over the parquet files.

- [x] ** Primary key uniqueness
  - validated uniqueness of `(item_id, store_id, date)`.
  - Ensures there is only one sales series per item-store-date combination.

- [x] **Date Continuity**
  - Validated that each `(item_id,store_id)` time series contains a continous daily date range.
  - Ensures that missing dates are not introduced during transformation.

- [x] **Sales Value Validation**
  - Checked for negative sales.
  - Expected conditon `sales >= 0`

- [x] **Selling Price Validation**
  - Checked for negative selling price.
  - Expected condition `sell_price > 0` when a price is present.

- [x] **Pre-Launch Price NULL Validation**
  - Identified the first recorded price week for each `(item_id,store_id)` combination.
  - NULL `sell_price` before the recorded first launch date is expected.

- [x] **Post-Launch Price NULL Validation**
  - Identified the first recorded price week for each `(item_id,store_id)` combination.
  - NULL `sell_price` after the recorded first launch date is treated as potential data qualiy issue.