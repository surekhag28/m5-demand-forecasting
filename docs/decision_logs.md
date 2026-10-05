# Decision Logs During Analysis and Modelling

This document captures the key issues, decisions, and trade-offs made during data preparation, feature engineering, baseline modelling, and model evaluation for the M5 Retail Demand Forecasting project.

---

## Issue 1: Understanding the Source Datasets

The M5 dataset consists of three primary datasets, each serving a different purpose.

### Sales

* The historical sales data is provided in **wide format**, where columns such as `d_1`, `d_2`, etc. represent individual days.
* Each **item-store combination represents an individual time series**.
* The wide format is not ideal for most time-series feature engineering, so I transformed the data into **long format**, where each row represents an item-store-date combination.

### Calendar

The `calendar` dataset acts as a lookup table that maps the `d_*` identifiers to actual dates and provides additional information such as:

* Weekday
* Month and year
* SNAP indicators
* Event information

### Sell Prices

The `sell_prices` dataset contains the **weekly selling price** for each item-store combination.

Understanding how these datasets relate to each other was important before designing the transformation and feature engineering steps.

---

## Decision 1: Reduce Memory Usage During Ingestion

The M5 datasets are relatively large, so I considered memory usage from the beginning rather than loading everything using the default Pandas data types.

For the `calendar` and `sell_prices` datasets:

* Numerical columns were **downcasted** where appropriate to smaller types such as `int8`, `uint8`, or `int32`, depending on the value range.
* String/object columns with a limited number of unique values were converted to **Pandas categorical types**.

Categorical types helped reduce memory usage because repeated string values do not need to be stored as separate Python objects for every row. Instead, Pandas maintains the unique values separately and stores compact numerical codes for the observations.

> **Note:** Detailed data analysis and memory checks are documented in [`docs/data_checks_logs.md`](data_checks_logs.md).

---

## Issue 2: Sales Dataset Size After Transformation

The original sales dataset is very wide. After melting it from wide to long format, it expands to **58M+ item-store-date observations**.

Materialising the complete transformed dataset as a Pandas DataFrame requires substantial RAM and made the ingestion process unstable on a local machine.

### Decision 2: Process the Sales Dataset in Chunks

Instead of loading and transforming the entire sales dataset at once:

1. I read the original sales data in **chunks before melting**.
2. Transform each chunk from wide to long format.
3. Join each chunk with the `calendar` dataset to:

   * Replace the `d_*` identifiers with actual dates.
   * Add calendar-related information.
4. Join the chunk with `sell_prices` to add the relevant price information.
5. Write each processed chunk to a **separate Parquet file**.

This kept the amount of data held in memory at any one time manageable while still allowing the complete dataset to be processed.

### Decision 3: Add Data Quality Checks

Data quality checks were performed at both the:

* **Bronze / raw layer**
* **Silver / processed layer**

These checks were used to validate the source data and transformations before moving into feature engineering.

> **Note:** Detailed data quality checks are documented in [`docs/data_quality_checks.md`](data_quality_checks.md).

---

## Issue 3: Pandas Memory Limitation During Feature Engineering

The final long-format sales dataset contains **58M+ observations**.

Building candidate features directly with Pandas required materialising the dataset into memory. During feature engineering, memory consumption became too high and the local environment eventually ran out of RAM.

This made it impractical to perform all candidate feature engineering directly with Pandas.

### Decision 4: Use DuckDB for Candidate Feature Engineering

To address the memory limitation, I used **DuckDB locally** to read the Parquet data and generate candidate features.

Instead of materialising the complete 58M+ rows as a Pandas DataFrame:

* DuckDB reads the Parquet data through its analytical query engine.
* Feature transformations are expressed as relational operations and SQL queries.
* DuckDB processes data in vectors/batches rather than creating Python objects for every row.
* Large intermediate datasets can remain within DuckDB instead of repeatedly being converted into Pandas DataFrames.

The feature engineering pipeline was divided into **modular feature groups**, allowing each group to progressively transform the existing DuckDB relation.

The resulting feature datasets were then written to the **Gold layer as Parquet files**.

### Candidate Feature Creation

The M5 demand forecasting problem can be viewed as addressing two related questions:

1. **Is the item likely to have a sale?**
2. **If a sale occurs, how much is likely to be sold?**

Based on this, candidate features were grouped around:

* Product lifecycle
* Calendar and seasonality
* Sales likelihood / intermittency
* Sales demand magnitude
* Recent demand
* Recent demand level

> **Note:** Detailed feature definitions and rationale are documented in [`docs/feature_decisions.md`](feature_decisions.md).

### Alternative Consideration: PySpark

PySpark was also considered as an alternative for candidate feature engineering, particularly for a future cloud or distributed environment.

PySpark could provide:

* Distributed processing across multiple workers
* Parallel execution
* Lazy execution
* The ability to scale beyond the resources of a local machine

However, the M5 feature engineering workload relies heavily on operations such as sorting, partitioning, window functions, and group transformations. These operations can involve significant data movement and shuffling.

For the current local setup, DuckDB provided a simpler approach while still addressing the memory limitations. PySpark can be evaluated later if the pipeline moves to a distributed or cloud environment.

---

## Issue 4: Loading Data for Baseline and ML Model Training

For model training, I initially loaded the entire candidate feature dataset from Parquet into memory. This took more than **72 seconds** and created a high memory load.

### Decision 5: Reduce Data Loaded During Model Training

To reduce both processing time and memory usage:

* I changed the approach to load only the training data required for each cross-validation fold rather than loading the held-out validation data unnecessarily.
* Features were downcasted to more appropriate data types where possible to reduce the memory footprint.
* When creating candidate features, I focused on a more recent sales history rather than using the entire historical dataset, as recent demand patterns are likely to have more influence on future demand.
* For the baseline models, I loaded only the columns required by each model instead of reading all candidate features.

### Decision 6: Use LightGBM for the ML Model

I chose **LightGBM** because it is fast and efficient for large datasets with a relatively high number of features.

I used LightGBM's `Dataset` structure for the training and validation folds. This allows LightGBM to efficiently represent the data and bin feature values, helping reduce memory and computational overhead during training.

### Decision 7: Reduce Cross-Validation from 5 Folds to 3 Folds

I initially considered 5 cross-validation folds but reduced this to 3 folds to balance model validation with computational cost.

Three folds provided sufficient variation across the training and validation periods while reducing the overall model training time.


----

## Issue 5: Train/Serve feature mismatch during training

So the task is to forecast sales for the next 28 days for every item-store combo.

For features I made things like lag_28, lag_36 and so on. On the training data these work fine since they just pull from past sales that we actually have.

The problem came up with the validation set. What I did was take the lag_28, lag_36 values from the last training row of each item and just copy them across all 28 days of validation. That **causes a train/serve mismatch**.

Here's why. In training, every row gets a lag_28 that's exactly 28 days back, so it's always fresh. But in validation, every row is stuck with the same frozen value from that last training day. For day 1 of validation that's fine, but by day 27 the model is looking at a value that's way older than what it saw during training.

And lag_28 on the last training day is already a month old. Sales can change a lot month to month, so if we reuse that same number for the whole 28-day window, the later days end up getting info that's almost two months old. Basically we're feeding the model stale numbers and it's going to make worse predictions because of it.

### Decision 8: Removing train/server feature mistmatch

* I noticed this mismatch while going through the analysis, so I went back and changed how the features are built.
* I set 28 as the smallest lag allowed and made lag_28, lag_35 and lag_42 for the whole dataset at once. Now every validation row gets its own correct past values, just like the training rows do, instead of a copied number.
* For the rolling features, I first shifted back 28 days and then took the 7, 14 and 28 day means ending at that point. That way none of them use sales from inside the forecast window, so no data leakage.


---

## Issue 6: Data Type Overflow During WRMSSE Evaluation

During data ingestion, I downcast `sales` to `uint8` and `sell_price` to `float16` to reduce memory usage during storage and processing.

However, I encountered a numerical overflow issue during WRMSSE evaluation.

WRMSSE is calculated across multiple aggregation levels. At higher levels, such as total, state, and store, daily sales values can become much larger after aggregation.

`float16` can represent values only up to approximately **65,504**. When aggregated values were squared as part of the RMSSE calculation, some values exceeded this limit and overflowed to `inf`.

This caused the scale calculation to become `inf`, which then resulted in RMSSE values becoming `0` or `NaN`. As a result, those aggregation levels were effectively contributing nothing to the final WRMSSE score.

I confirmed the issue by comparing the results before and after changing the data type. With `float16`, the total and state aggregation levels dropped from around **1.1 to exactly 0**, which was clearly incorrect.

### Decision 8: Use `float64` for WRMSSE Evaluation

To avoid numerical overflow, I convert both `sales` and `sell_price` to `float64` in the training and validation datasets before calculating the WRMSSE components:

* Scale
* Weights
* RMSSE

This allows me to keep memory-efficient data types during ingestion and feature processing while using a safer numerical type for the final evaluation calculations.

---

## Issue 7: LightGBM Baseline With All Candidate Features

For the initial LightGBM baseline, I used all candidate features during cross-validation.

This improved the WRMSSE score compared with both the naive and seasonal naive baseline models.

However, using all candidate features in the final model could introduce redundant or less useful features. I therefore wanted to identify which features or feature groups were actually contributing to the model.

### Decision 9: Use Permutation Validation for Feature Selection

I used permutation validation to assess the contribution of different feature groups.

* I grouped the candidate features into feature groups based on the type of information they represent.
* I added a random **noise feature** to provide a reference threshold when deciding whether a feature group was providing meaningful information.
* I trained the model using cross-validation and, for each validation fold, shuffled the values of one feature group at a time.
* After shuffling a group, I calculated the **delta in WRMSSE** by comparing the shuffled score with the original validation WRMSSE. This showed how much model performance changed when the information from that feature group was disrupted.
* I calculated the delta for each fold and then took the mean delta across the folds to compare the contribution of different feature groups.

A larger positive increase in WRMSSE after shuffling indicates that the model was relying more heavily on that feature group. A delta close to the noise feature suggests that the group may not be providing much useful signal.

### Decision 10: Perform Permutation Testing During the Existing CV Run

* Rather than running another complete cross-validation cycle specifically for feature selection, I calculated the permutation deltas during the same CV run used for the initial model evaluation.
* This reduced the overall computation time while giving me an early understanding of which feature groups were contributing most to the model.
* To make training faster, I only used the last 2 years of sales data. I also cut down the number of trees, used fewer bins when the model looks for split points, and increased the learning rate.
* Permutation validation was also simpler and faster than approaches such as forward selection or backward elimination because it does not require retraining the model for every possible feature combination.
* 
