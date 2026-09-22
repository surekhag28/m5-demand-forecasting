## Decision Logs During Analysis and Modelling

### Observation 1 — Understanding the source datasets

- The historical sales dataset is provided in a **wide format**, where columns such as `d_1`, `d_2`, etc. represent different dates.
- Each **item-store combination represents an individual time series**.
- The `calendar` dataset acts as a lookup table that maps the `d_*` identifiers to actual dates and provides additional information such as:
  - weekday
  - month/year
  - SNAP information
  - event information
- The `sell_prices` dataset contains the **weekly selling price** for each item-store combination.

### Decision — Reduce memory usage during ingestion

- After loading the `calendar` and `sell_prices` datasets, I **downcasted data types** where possible to reduce memory usage during processing.
- Numerical columns were downcasted to smaller types such as `int8`, `uint8`, or `int32`, depending on the range of values in each column.
- String/object columns with a limited number of unique values were converted to **Pandas categorical types**.
  - Pandas `object` columns can have significant memory overhead because they store references to Python string objects.
  - Categorical columns instead maintain a dictionary of unique values and store compact numerical codes in the actual column.
  - This significantly reduced the memory required for repeated categorical values.

> **Note:** Detailed data analysis and memory checks are documented in `@docs/data_checks_logs.md`.

### Observation 1.1 — Sales dataset size after transformation

- The original sales dataset is very wide.
- After melting it from wide to long format, it expands to **58M+ item-store-date observations**.
- Loading the complete transformed dataset into a Pandas DataFrame would require a large amount of RAM and could make the ingestion process unstable.

### Decision — Process the sales dataset in chunks

- Instead of loading the complete sales dataset into memory:
  - I read the original sales data **in chunks before melting**.
  - Each chunk was melted from wide to long format.
  - The transformed chunk was joined with the `calendar` dataset to:
    - replace the `d_*` identifiers with actual dates
    - add calendar-related information
  - The chunk was then joined with `sell_prices` to add the relevant price information.
- Each processed chunk was written to a **separate Parquet file** rather than creating one large file.
- This allowed the ingestion process to work with a manageable amount of data in memory at any given time.

### Decision — Add data quality checks

- I performed data quality checks at both:
  - **Bronze/raw layer**
  - **Silver/processed layer**
- These checks were used to validate the transformed data before moving further into feature engineering.

> **Note:** Detailed data quality checks are documented in `@docs/data_quality_checks.md`.

---

### Observation 2 — Pandas memory limitation during feature engineering

- The final long-format sales dataset contains **58M+ observations**.
- Building candidate features directly with Pandas required loading the complete dataset into memory.
- During feature engineering, the memory usage became too high and the local environment eventually ran out of RAM.

### Decision — Use DuckDB for candidate feature engineering

- To address the memory limitation, I used **DuckDB locally** to read the Parquet data and build the candidate features.
- Instead of materialising the complete 58M+ rows as a Pandas DataFrame:
  - DuckDB reads the Parquet data through its analytical query engine.
  - Feature transformations are expressed as relational operations.
  - DuckDB can process the data in vectors/batches rather than creating Python objects for every row.
- I created **modular functions for each candidate feature group**.
- Each function takes the existing DuckDB relation and projects the columns required for that feature group.
- This allowed the feature-engineering pipeline to be built incrementally while keeping the data in DuckDB rather than repeatedly converting large datasets into Pandas DataFrames.

### Result

- Candidate features were generated using DuckDB without requiring the complete 58M+ row dataset to be materialised in Pandas memory.
- The resulting feature dataset was written back to the **Gold layer in Parquet format** for downstream modelling and evaluation.


### Observation 3 — Memory limitation during baseline modelling

- For baseline model evaluation, loading the complete **58M+ row Gold dataset** into a Pandas DataFrame was again causing significant memory usage.
- However, the naive and seasonal naive baselines do not require all columns or the full historical dataset.

### Decision — Load only the data required for baseline models

- For the baseline and seasonal naive baseline models, I first identified the minimum data required:
  - `item_id`
  - `store_id`
  - `date`
  - `sales`
- Instead of loading the complete dataset:
  - I selected only these required columns.
  - I filtered the data to the **recent historical period needed to generate the baseline predictions** for each item-store combination.
- This avoided loading unnecessary features and older historical data into the Pandas DataFrame.
- As a result, the amount of data materialised in memory was significantly reduced while still providing everything required to calculate the baseline predictions.


