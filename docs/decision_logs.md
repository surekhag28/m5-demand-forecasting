# Decision Logs During Analysis and Modelling

This document captures the key issues, decisions, and trade-offs made during data preparation, feature engineering, baseline modelling, and model evaluation for the **M5 Retail Demand Forecasting** project.

## Table of Contents

- [Decision Logs During Analysis and Modelling](#decision-logs-during-analysis-and-modelling)
  - [Table of Contents](#table-of-contents)
  - [Issue 1: Understanding the Source Datasets](#issue-1-understanding-the-source-datasets)
    - [Sales](#sales)
    - [Calendar](#calendar)
    - [Sell Prices](#sell-prices)
    - [Decision 1: Reduce Memory Usage During Ingestion](#decision-1-reduce-memory-usage-during-ingestion)
  - [Issue 2: Sales Dataset Size After Transformation](#issue-2-sales-dataset-size-after-transformation)
    - [Decision 2: Process the Sales Dataset in Chunks](#decision-2-process-the-sales-dataset-in-chunks)
    - [Decision 3: Add Data Quality Checks](#decision-3-add-data-quality-checks)
  - [Issue 3: Pandas Memory Limitation During Feature Engineering](#issue-3-pandas-memory-limitation-during-feature-engineering)
    - [Decision 4: Use DuckDB for Candidate Feature Engineering](#decision-4-use-duckdb-for-candidate-feature-engineering)
      - [Candidate Feature Creation](#candidate-feature-creation)
      - [Alternative Considered: PySpark](#alternative-considered-pyspark)
  - [Issue 4: Loading Data for Baseline and ML Model Training](#issue-4-loading-data-for-baseline-and-ml-model-training)
    - [Decision 5: Reduce Data Loaded During Model Training](#decision-5-reduce-data-loaded-during-model-training)
    - [Decision 6: Use LightGBM for the ML Model](#decision-6-use-lightgbm-for-the-ml-model)
    - [Decision 7: Reduce Cross-Validation from 5 Folds to 3 Folds](#decision-7-reduce-cross-validation-from-5-folds-to-3-folds)
  - [Issue 5: Train/Serve Feature Mismatch During Training](#issue-5-trainserve-feature-mismatch-during-training)
    - [Decision 8: Remove the Train/Serve Feature Mismatch](#decision-8-remove-the-trainserve-feature-mismatch)
  - [Issue 6: Data Type Overflow During WRMSSE Evaluation](#issue-6-data-type-overflow-during-wrmsse-evaluation)
    - [Decision 9: Use `float64` for WRMSSE Evaluation](#decision-9-use-float64-for-wrmsse-evaluation)
  - [Issue 7: LightGBM Baseline With All Candidate Features](#issue-7-lightgbm-baseline-with-all-candidate-features)
    - [Decision 10: Use the Feature Drop Validation Technique](#decision-10-use-the-feature-drop-validation-technique)
      - [Second Iteration](#second-iteration)
    - [Decision 11: Identify the Best Number of Trees](#decision-11-identify-the-best-number-of-trees)
      - [Global LightGBM Model](#global-lightgbm-model)
      - [Store-Specific LightGBM Models](#store-specific-lightgbm-models)
      - [Ensembling (Global + Store-Specific Models)](#ensembling-global--store-specific-models)
  - [Error Analysis](#error-analysis)
    - [Error Analysis by Store](#error-analysis-by-store)
      - [Key Observations](#key-observations)
    - [Error Analysis by Category](#error-analysis-by-category)
      - [Key Observations](#key-observations-1)
    - [Error Analysis by Sales Frequency](#error-analysis-by-sales-frequency)
      - [Key Observations](#key-observations-2)
    - [WRMSSE by Store](#wrmsse-by-store)
    - [WRMSSE by Category](#wrmsse-by-category)
    - [WRMSSE by Price Band](#wrmsse-by-price-band)
    - [WRMSSE by Sales Frequency](#wrmsse-by-sales-frequency)

---

## Issue 1: Understanding the Source Datasets

The M5 dataset consists of three primary datasets, each serving a different purpose.

### Sales

- The historical sales data is provided in **wide format**, where columns such as `d_1`, `d_2`, etc. represent individual days.
- Each **item-store combination represents an individual time series**.
- The wide format is not ideal for most time-series feature engineering, so I transformed the data into **long format**, where each row represents an item-store-date combination.

### Calendar

The `calendar` dataset acts as a lookup table that maps the `d_*` identifiers to actual dates and provides additional information such as:

- Weekday
- Month and year
- SNAP indicators
- Event information

### Sell Prices

The `sell_prices` dataset contains the **weekly selling price** for each item-store combination.

Understanding how these datasets relate to each other was important before designing the transformation and feature engineering steps.

### Decision 1: Reduce Memory Usage During Ingestion

The M5 datasets are relatively large, so I considered memory usage from the beginning rather than loading everything using the default Pandas data types.

For the `calendar` and `sell_prices` datasets:

- Numerical columns were **downcast** where appropriate to smaller types such as `int8`, `uint8`, or `int32`, depending on the value range.
- String/object columns with a limited number of unique values were converted to **Pandas categorical types**.

Categorical types helped reduce memory usage because repeated string values do not need to be stored as separate Python objects for every row. Instead, Pandas maintains the unique values separately and stores compact numerical codes for the observations.

> [!NOTE]
> Detailed data analysis and memory checks are documented in [`docs/data_checks_logs.md`](data_checks_logs.md).

---

## Issue 2: Sales Dataset Size After Transformation

The original sales dataset is very wide. After melting it from wide to long format, it expands to **58M+ item-store-date observations**.

Materialising the complete transformed dataset as a Pandas DataFrame requires substantial RAM and made the ingestion process unstable on a local machine.

### Decision 2: Process the Sales Dataset in Chunks

Instead of loading and transforming the entire sales dataset at once:

1. Read the original sales data in **chunks before melting**.
2. Transform each chunk from wide to long format.
3. Join each chunk with the `calendar` dataset to:
   - Replace the `d_*` identifiers with actual dates.
   - Add calendar-related information.
4. Join the chunk with `sell_prices` to add the relevant price information.
5. Write each processed chunk to a **separate Parquet file**.

This kept the amount of data held in memory at any one time manageable while still allowing the complete dataset to be processed.

### Decision 3: Add Data Quality Checks

Data quality checks were performed at both the:

- **Bronze / raw layer**
- **Silver / processed layer**

These checks were used to validate the source data and transformations before moving into feature engineering.

> [!NOTE]
> Detailed data quality checks are documented in [`docs/data_quality_checks.md`](data_quality_checks.md).

---

## Issue 3: Pandas Memory Limitation During Feature Engineering

The final long-format sales dataset contains **58M+ observations**.

Building candidate features directly with Pandas required materialising the dataset into memory. During feature engineering, memory consumption became too high and the local environment eventually ran out of RAM.

This made it impractical to perform all candidate feature engineering directly with Pandas.

### Decision 4: Use DuckDB for Candidate Feature Engineering

To address the memory limitation, I used **DuckDB locally** to read the Parquet data and generate candidate features.

Instead of materialising the complete 58M+ rows as a Pandas DataFrame:

- DuckDB reads the Parquet data through its analytical query engine.
- Feature transformations are expressed as relational operations and SQL queries.
- DuckDB processes data in vectors/batches rather than creating Python objects for every row.
- Large intermediate datasets can remain within DuckDB instead of repeatedly being converted into Pandas DataFrames.

The feature engineering pipeline was divided into **modular feature groups**, allowing each group to progressively transform the existing DuckDB relation.

The resulting feature datasets were then written to the **Gold layer as Parquet files**.

#### Candidate Feature Creation

The M5 demand forecasting problem can be viewed as addressing two related questions:

1. **Is the item likely to have a sale?**
2. **If a sale occurs, how much is likely to be sold?**

Based on this, candidate features were grouped around:

- Product lifecycle
- Calendar and seasonality
- Sales likelihood / intermittency
- Sales demand magnitude
- Recent demand
- Recent demand level

> [!NOTE]
> Detailed feature definitions and rationale are documented in [`docs/feature_decisions.md`](feature_decisions.md).

#### Alternative Considered: PySpark

PySpark was also considered as an alternative for candidate feature engineering, particularly for a future cloud or distributed environment.

PySpark could provide:

- Distributed processing across multiple workers
- Parallel execution
- Lazy execution
- The ability to scale beyond the resources of a local machine

However, the M5 feature engineering workload relies heavily on operations such as sorting, partitioning, window functions, and group transformations. These operations can involve significant data movement and shuffling.

For the current local setup, DuckDB provided a simpler approach while still addressing the memory limitations. PySpark can be evaluated later if the pipeline moves to a distributed or cloud environment.

---

## Issue 4: Loading Data for Baseline and ML Model Training

For model training, I initially loaded the entire candidate feature dataset from Parquet into memory. This took more than **72 seconds** and created a high memory load.

### Decision 5: Reduce Data Loaded During Model Training

To reduce both processing time and memory usage:

- I changed the approach to load only the training data required for each cross-validation fold, rather than loading the held-out validation data unnecessarily.
- Features were downcast to more appropriate data types where possible to reduce the memory footprint.
- When creating candidate features, I focused on more recent sales history rather than the entire historical dataset, as recent demand patterns are likely to have more influence on future demand.
- For the baseline models, I loaded only the columns required by each model instead of reading all candidate features.

### Decision 6: Use LightGBM for the ML Model

I chose **LightGBM** because it is fast and efficient for large datasets with a relatively high number of features.

I used LightGBM's `Dataset` structure for the training and validation folds. This allows LightGBM to efficiently represent the data and bin feature values, helping reduce memory and computational overhead during training.

### Decision 7: Reduce Cross-Validation from 5 Folds to 3 Folds

I initially considered 5 cross-validation folds but reduced this to 3 folds to balance model validation with computational cost.

Three folds provided sufficient variation across the training and validation periods while reducing the overall model training time.

---

## Issue 5: Train/Serve Feature Mismatch During Training

The task is to forecast sales for the **next 28 days** for every item-store combination.

For features, I created lags such as `lag_28`, `lag_36`, and so on. On the training data these work fine, since they pull from past sales that are actually available.

The problem came up with the validation set. I took the `lag_28` and `lag_36` values from the **last training row** of each item and copied them across all 28 days of validation. This **causes a train/serve mismatch**.

**Why this is a problem:**

- In training, every row gets a `lag_28` that is exactly 28 days back, so it is always fresh.
- In validation, every row is stuck with the same frozen value from the last training day.
- For day 1 of validation this is fine, but by day 27 the model is looking at a value much older than anything it saw during training.
- `lag_28` on the last training day is already a month old. Reusing that number for the whole 28-day window means the later days get information that is almost **two months old**.

Sales can change a lot month to month, so the model is fed stale numbers and makes worse predictions as a result.

### Decision 8: Remove the Train/Serve Feature Mismatch

- I noticed this mismatch during the analysis, so I went back and changed how the features are built.
- I set **28 as the smallest allowed lag** and created `lag_28`, `lag_35`, and `lag_42` for the **whole dataset at once**. Now every validation row gets its own correct past values, just like the training rows, instead of a copied number.
- For the rolling features, I first **shifted back 28 days** and then took the 7, 14, and 28-day means ending at that point. This way none of them use sales from inside the forecast window, so there is **no data leakage**.

---

## Issue 6: Data Type Overflow During WRMSSE Evaluation

During data ingestion, I downcast `sales` to `uint8` and `sell_price` to `float16` to reduce memory usage during storage and processing.

However, I encountered a **numerical overflow** issue during WRMSSE evaluation.

WRMSSE is calculated across multiple aggregation levels. At higher levels, such as total, state, and store, daily sales values become much larger after aggregation.

`float16` can only represent values up to approximately **65,504**. When aggregated values were squared as part of the RMSSE calculation, some values exceeded this limit and overflowed to `inf`.

This caused the scale calculation to become `inf`, which then resulted in RMSSE values becoming `0` or `NaN`. As a result, those aggregation levels were effectively contributing nothing to the final WRMSSE score.

I confirmed the issue by comparing results before and after changing the data type. With `float16`, the total and state aggregation levels dropped from around **1.1 to exactly 0**, which was clearly incorrect.

### Decision 9: Use `float64` for WRMSSE Evaluation

To avoid numerical overflow, I convert both `sales` and `sell_price` to `float64` in the training and validation datasets before calculating the WRMSSE components:

- Scale
- Weights
- RMSSE

This keeps memory-efficient data types during ingestion and feature processing, while using a safer numerical type for the final evaluation calculations.

---

## Issue 7: LightGBM Baseline With All Candidate Features

For the forecasting model, I used a **non-recursive (direct) approach** to predict the full 28-day window instead of a recursive one. The trade-off is losing short, recent lags such as `lag_1`, `lag_2`, and `lag_7`.

A **recursive approach** was the alternative. It would allow short lags and the most recent sales to be used to forecast the next day. However, each prediction then becomes an input for the next one, so if an early forecast is off, the error **compounds across the window**. This felt riskier, since the model would essentially be fed incorrect information about past sales.

For the initial LightGBM baseline, I used all candidate features during cross-validation. This improved the WRMSSE score compared with both the naive and seasonal naive baselines:

| Model                     | WRMSSE   |
|---------------------------|---------:|
| Naive baseline            | 1.4638   |
| Seasonal naive baseline   | 0.8690   |
| LightGBM + all features   | 0.629043 |

However, using all candidate features in the final model could introduce redundant or less useful features. I therefore wanted to identify which features or feature groups were actually contributing to the model.

### Decision 10: Use the Feature Drop Validation Technique

I used the **feature drop validation** technique to quickly check which features likely impact model performance. In every cross-validation fold, I dropped one feature group at a time and observed the WRMSSE score across the folds.

**LightGBM baseline WRMSSE = `0.629043`**

| Model                   |   Fold 1 |   Fold 2 |   Fold 3 | Mean WRMSSE | Mean Delta | Folds Worse |
|-------------------------|---------:|---------:|---------:|------------:|-----------:|------------:|
| baseline                | 0.689039 | 0.681166 | 0.516924 |    0.629043 |   0.000000 |           0 |
| enc_item_store_dropped  | 0.694783 | 0.688421 | 0.566918 |    0.650041 |   0.020998 |           3 |
| events_dropped          | 0.697881 | 0.699155 | 0.540617 |    0.645884 |   0.016841 |           3 |
| ids_dropped             | 0.666442 | 0.661149 | 0.514679 |    0.614090 |  -0.014953 |           0 |
| intermittency_dropped   | 0.686140 | 0.689864 | 0.526902 |    0.634302 |   0.005259 |           2 |
| lags_dropped            | 0.690365 | 0.687401 | 0.516593 |    0.631453 |   0.002410 |           2 |
| price_change_dropped    | 0.696253 | 0.682279 | 0.518516 |    0.632349 |   0.003306 |           3 |
| price_level_dropped     | 0.687979 | 0.691624 | 0.533872 |    0.637825 |   0.008782 |           2 |
| rolling_mean_dropped    | 0.706299 | 0.672092 | 0.526439 |    0.634943 |   0.005900 |           2 |
| seasonality_dropped     | 0.730048 | 0.603620 | 0.523050 |    0.618906 |  -0.010137 |           2 |
| snap_dropped            | 0.677893 | 0.678365 | 0.531658 |    0.629305 |   0.000262 |           1 |
| trends_position_dropped | 0.696758 | 0.688395 | 0.509915 |    0.631689 |   0.002646 |           2 |
| weekly_pattern_dropped  | 0.929121 | 0.907045 | 0.921591 |    0.919252 |   0.290209 |           3 |

**Observation:** After removing the `ids` feature group, the WRMSSE score decreased across all folds compared to the baseline, indicating an improvement in forecasting performance. Since the encoding features already capture information about entities such as `state_id`, `store_id`, and `item_id`, the raw ID features may provide redundant information or add unnecessary complexity. Therefore, I decided to **drop the `ids` group**.

#### Second Iteration

After dropping the `ids` features (`year`, `state_id`, `store_id`, `cat_id`, `dept_id`, and `item_id`) and splitting seasonality into **day of month**, **month**, and **week**:

| Model                     |   Fold 1 |   Fold 2 |   Fold 3 | Mean WRMSSE | Mean Delta | Folds Worse |
|---------------------------|---------:|---------:|---------:|------------:|-----------:|------------:|
| baseline                  | 0.666442 | 0.661149 | 0.514679 |    0.614090 |   0.000000 |           0 |
| enc_item_store_dropped    | 0.704818 | 0.708249 | 0.579123 |    0.664063 |   0.049973 |           3 |
| events_dropped            | 0.684380 | 0.696640 | 0.531460 |    0.637493 |   0.023403 |           3 |
| intermittency_dropped     | 0.676599 | 0.653937 | 0.521163 |    0.617233 |   0.003142 |           2 |
| lags_dropped              | 0.688286 | 0.676748 | 0.561885 |    0.642306 |   0.028216 |           3 |
| price_change_dropped      | 0.665185 | 0.655020 | 0.522206 |    0.614137 |   0.000047 |           1 |
| price_level_dropped       | 0.660096 | 0.647582 | 0.512595 |    0.606758 |  -0.007333 |           0 |
| seasonality_dom_dropped   | 0.648262 | 0.669426 | 0.516018 |    0.611235 |  -0.002855 |           2 |
| seasonality_month_dropped | 0.685715 | 0.669240 | 0.514392 |    0.623116 |   0.009025 |           2 |
| seasonality_week_dropped  | 0.674890 | 0.674156 | 0.508472 |    0.619173 |   0.005082 |           2 |
| snap_dropped              | 0.681866 | 0.659106 | 0.513791 |    0.618254 |   0.004164 |           1 |
| trends_position_dropped   | 0.683638 | 0.658958 | 0.514199 |    0.618932 |   0.004841 |           1 |
| weekly_pattern_dropped    | 0.916358 | 0.907156 | 0.943238 |    0.922251 |   0.308160 |           3 |

### Decision 11: Identify the Best Number of Trees

#### Global LightGBM Model

I wanted to identify the best number of trees for better forecasts while reducing overfitting.

```python
TREE_GRID = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
```

| Trees | Fold 1 | Fold 2 | Fold 3 | Mean WRMSSE |
|------:|-------:|-------:|-------:|------------:|
|   100 | 0.6751 | 0.6358 | 0.5196 |      0.6102 |
|   200 | 0.6541 | 0.6369 | 0.5079 |      0.5996 |
|   300 | 0.6569 | 0.6383 | 0.5027 |      0.5993 |
|   400 | 0.6572 | 0.6400 | 0.4993 |      0.5988 |
|   500 | 0.6559 | 0.6401 | 0.4993 |      0.5984 |
|   750 | 0.6624 | 0.6458 | 0.4988 |      0.6023 |
|  1000 | 0.6701 | 0.6473 | 0.5076 |      0.6084 |
|  1500 | 0.6727 | 0.6558 | 0.5100 |      0.6128 |
|  2000 | 0.6764 | 0.6624 | 0.5124 |      0.6171 |

**Best tree count (lowest mean WRMSSE):** `500`

| Model    |   Fold 1 |   Fold 2 |   Fold 3 | Mean WRMSSE | Mean Delta | Folds Worse |
|----------|---------:|---------:|---------:|------------:|-----------:|------------:|
| baseline | 0.676429 | 0.662429 | 0.512352 |    0.617070 |   0.000000 |           0 |

**Chosen `num_iterations`:** `400`

- After evaluating different numbers of estimators through cross-validation, **400 trees** emerged as the best choice. This configuration achieved a lower WRMSSE across the validation folds and the lowest mean WRMSSE compared to the other configurations tested.
- Based on these results, I trained the final global LightGBM model using 400 trees.

```text
Global model test WRMSSE (400 trees): 0.5929
```

#### Store-Specific LightGBM Models

I also developed **10 store-specific LightGBM models** to investigate whether individual models could better capture store-level patterns and demand behaviours that the global model might overlook.

For each store-specific model, I evaluated different numbers of estimators using cross-validation and selected the one with the lowest mean WRMSSE.

**First grid:**

```python
TREE_GRID = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
```

| Trees | Fold 1 | Fold 2 | Fold 3 | Mean WRMSSE |
|------:|-------:|-------:|-------:|------------:|
|   250 | 0.6411 | 0.5716 | 0.4830 |      0.5653 |
|   500 | 0.6480 | 0.5781 | 0.4841 |      0.5701 |
|   750 | 0.6577 | 0.5870 | 0.4868 |      0.5772 |
|  1000 | 0.6643 | 0.5942 | 0.4893 |      0.5826 |
|  1250 | 0.6724 | 0.6001 | 0.4923 |      0.5883 |
|  1500 | 0.6814 | 0.6058 | 0.4931 |      0.5934 |
|  1750 | 0.6868 | 0.6107 | 0.4935 |      0.5970 |
|  2000 | 0.6927 | 0.6160 | 0.4955 |      0.6014 |

Since the lowest value was at the smallest tree count, I searched a lower range.

**Second grid:**

```python
TREE_GRID = [50, 100, 150, 200, 250, 300, 400]
```

| Trees | Fold 1 | Fold 2 | Fold 3 | Mean WRMSSE |
|------:|-------:|-------:|-------:|------------:|
|    50 | 0.7741 | 0.6882 | 0.6309 |      0.6977 |
|   100 | 0.6689 | 0.5806 | 0.5103 |      0.5866 |
|   150 | 0.6484 | 0.5680 | 0.4877 |      0.5680 |
|   200 | 0.6410 | 0.5689 | 0.4850 |      0.5650 |
|   250 | 0.6411 | 0.5716 | 0.4830 |      0.5653 |
|   300 | 0.6426 | 0.5727 | 0.4830 |      0.5661 |
|   400 | 0.6426 | 0.5765 | 0.4842 |      0.5678 |

Based on the cross-validation results, **200 estimators** emerged as the best choice for the store-specific models, achieving the lowest mean WRMSSE.

**After final training:**

| Metric                     | WRMSSE |
|----------------------------|-------:|
| CV mean (200 trees)        | 0.5650 |
| Test (`d_1914`–`d_1941`)   | 0.5658 |
| Difference                 | 0.0008 |

#### Ensembling (Global + Store-Specific Models)

| Model                     | Test WRMSSE |
|---------------------------|------------:|
| Global only               |      0.5934 |
| Store-specific only       |      0.5665 |
| Average of both           |      0.5720 |

---

## Error Analysis

### Error Analysis by Store

The store-level analysis compares forecasting performance across the ten stores using mean RMSSE, median RMSSE, and the percentage of item-store series with RMSSE greater than 1.

| Store | Number of Series | Mean RMSSE | Median RMSSE | Series with RMSSE > 1 (%) | Share of Series (%) |
|:---|---:|---:|---:|---:|---:|
| CA_2 | 3,048 | 0.8442 | 0.570 | 20.54% | 10.00% |
| WI_2 | 3,048 | 0.7526 | 0.530 | 17.75% | 10.00% |
| TX_3 | 3,046 | 0.7494 | 0.515 | 16.58% | 10.00% |
| CA_3 | 3,047 | 0.7313 | 0.510 | 14.77% | 10.00% |
| CA_4 | 3,046 | 0.7086 | 0.510 | 16.55% | 10.00% |
| WI_3 | 3,048 | 0.7303 | 0.510 | 17.19% | 10.00% |
| CA_1 | 3,048 | 0.8280 | 0.500 | 14.53% | 10.00% |
| WI_1 | 3,047 | 0.6744 | 0.480 | 13.29% | 10.00% |
| TX_2 | 3,048 | 0.6496 | 0.470 | 14.27% | 10.00% |
| TX_1 | 3,048 | 0.6945 | 0.460 | 14.73% | 10.00% |

#### Key Observations

- **CA_2 has the highest forecasting error**, with a mean RMSSE of 0.8442 and 20.54% of its series having RMSSE greater than 1. This makes it a priority for further investigation.
- **CA_1 also has a high mean RMSSE of 0.8280**, although its percentage of series with RMSSE greater than 1 is lower at 14.53%. This suggests that a smaller number of high-error series may be pulling its mean upward.
- **TX_2 has the lowest mean RMSSE**, at 0.6496, indicating better average forecasting performance than the other stores.
- **The median RMSSE is lower than the mean RMSSE for every store.** This suggests that higher-error series are increasing the average error across stores.
- Each store accounts for approximately 10% of the series, so differences in mean RMSSE are not simply explained by one store having substantially more series.

---

### Error Analysis by Category

The category-level analysis compares forecasting errors across FOODS, HOUSEHOLD, and HOBBIES.

| Category | Number of Series | Mean RMSSE | Median RMSSE | Series with RMSSE > 1 (%) | Share of Series (%) |
|:---|---:|---:|---:|---:|---:|
| FOODS | 14,366 | 0.7791 | 0.530 | 17.67% | 47.14% |
| HOUSEHOLD | 10,468 | 0.6773 | 0.490 | 14.19% | 34.35% |
| HOBBIES | 5,640 | 0.7367 | 0.480 | 15.23% | 18.51% |

#### Key Observations

- **FOODS has the highest mean RMSSE**, at 0.7791, and the highest proportion of series with RMSSE greater than 1, at 17.67%.
- **FOODS accounts for 47.14% of all item-store series**, so its larger share of high-error series may partly reflect its larger representation in the dataset.
- **HOUSEHOLD has the lowest mean RMSSE and the lowest percentage of series with RMSSE greater than 1**, suggesting better average forecasting performance.
- **HOBBIES has a higher mean RMSSE than HOUSEHOLD**, despite having a slightly lower median RMSSE. This suggests that higher-error series may be having a greater influence on its average.

### Error Analysis by Sales Frequency

This analysis examines how forecasting performance varies with sales intermittency, helping identify whether items with infrequent sales are more difficult to forecast than items with more regular demand.

| Sales Frequency | Number of Series | Mean RMSSE | Median RMSSE | Series with RMSSE > 1 (%) | Share of Series (%) |
|:---|---:|---:|---:|---:|---:|
| Low intermittency | 11,488 | 0.6911 | 0.520 | 13.78% | 37.70% |
| Highly intermittent | 3,774 | 1.0267 | 0.510 | 23.77% | 12.38% |
| Moderately intermittent | 15,212 | 0.6984 | 0.500 | 15.79% | 49.92% |

#### Key Observations

- **Highly intermittent series have the highest mean RMSSE**, at 1.0267, exceeding the RMSSE benchmark of 1. Approximately 23.77% of these series have RMSSE greater than 1, compared with 13.78% for low-intermittency series.
- **The mean RMSSE for highly intermittent series is much higher than its median RMSSE.** The mean is 1.0267, while the median is 0.51. This suggests that a subset of highly intermittent series has substantially higher errors, pulling up the average.
- **Moderately intermittent series represent the largest group**, accounting for 49.92% of all series. Their mean RMSSE is 0.6984, with 15.79% of series having RMSSE greater than 1.
- **Low-intermittency series have the lowest percentage of high-error series**, at 13.78%, and a mean RMSSE of 0.6911, indicating more consistent forecasting performance than the highly intermittent group.

### WRMSSE by Store

| Store | Number of Series | Mean RMSSE | Median RMSSE | Weighted Error Contribution | Total Weight | Share of Weighted Error |
| ----- | ---------------- | ---------- | ------------ | --------------------------- | ------------ | ----------------------- |
| CA_3  | 3,047            | 0.731280   | 0.510        | 0.120758                    | 0.155628     | 15.67%                  |
| WI_2  | 3,048            | 0.752618   | 0.530        | 0.095776                    | 0.116622     | 12.43%                  |
| CA_2  | 3,048            | 0.844242   | 0.570        | 0.093162                    | 0.110247     | 12.09%                  |
| CA_1  | 3,048            | 0.828005   | 0.500        | 0.081615                    | 0.110888     | 10.59%                  |
| TX_3  | 3,046            | 0.749448   | 0.515        | 0.076071                    | 0.096535     | 9.87%                   |
| TX_2  | 3,048            | 0.649580   | 0.470        | 0.067864                    | 0.095201     | 8.81%                   |
| WI_3  | 3,048            | 0.730338   | 0.510        | 0.067696                    | 0.084593     | 8.78%                   |
| WI_1  | 3,047            | 0.674365   | 0.480        | 0.060920                    | 0.087117     | 7.90%                   |
| TX_1  | 3,048            | 0.694455   | 0.460        | 0.060248                    | 0.077560     | 7.82%                   |
| CA_4  | 3,046            | 0.708601   | 0.510        | 0.046626                    | 0.065608     | 6.05%                   |

### WRMSSE by Category

| Category  | Number of Series | Mean RMSSE | Median RMSSE | Weighted Error Contribution | Total Weight | Share of Weighted Error |
| --------- | ---------------- | ---------- | ------------ | --------------------------- | ------------ | ----------------------- |
| FOODS     | 14,366           | 0.779132   | 0.530        | 0.464662                    | 0.568586     | 60.29%                  |
| HOUSEHOLD | 10,468           | 0.677294   | 0.490        | 0.224718                    | 0.303335     | 29.16%                  |
| HOBBIES   | 5,640            | 0.736699   | 0.480        | 0.081357                    | 0.128079     | 10.56%                  |

### WRMSSE by Price Band

| Price Band | Number of Series | Mean RMSSE | Median RMSSE | Weighted Error Contribution | Total Weight | Share of Weighted Error |
|:---|---:|---:|---:|---:|---:|---:|
| $2-5 | 14,712 | 0.733003 | 0.51 | 0.344372 | 0.427084 | 44.68% |
| $5-10 | 7,066 | 0.702228 | 0.51 | 0.213811 | 0.280457 | 27.74% |
| $1-2 | 4,679 | 0.794721 | 0.51 | 0.107383 | 0.137964 | 13.93% |
| $10+ | 1,810 | 0.810702 | 0.49 | 0.076594 | 0.115455 | 9.94% |
| <$1 | 2,207 | 0.682442 | 0.46 | 0.028577 | 0.039039 | 3.71% |


### WRMSSE by Sales Frequency

| Sales Frequency | Number of Series | Mean RMSSE | Median RMSSE | Weighted Error Contribution | Total Weight | Share of Weighted Error |
|:---|---:|---:|---:|---:|---:|---:|
| Low intermittency | 11,488 | 0.691135 | 0.52 | 0.561489 | 0.739618 | 72.85% |
| Moderately intermittent | 15,212 | 0.698353 | 0.50 | 0.194975 | 0.246394 | 25.30% |
| Highly intermittent | 3,774 | 1.026709 | 0.51 | 0.014272 | 0.013988 | 1.85% |




