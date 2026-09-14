# Decision Logs

Findings from data exploration and decisions made to handle the data efficiently during processing.

---

## Calendar Dataset

### Overview

The M5 `calendar` dataset contains **1,969 records and 14 columns**.

The initial data types are:

| Column         | Non-Null Count | Dtype   |
| -------------- | -------------: | ------- |
| `date`         |          1,969 | `str`   |
| `wm_yr_wk`     |          1,969 | `int64` |
| `weekday`      |          1,969 | `str`   |
| `wday`         |          1,969 | `int64` |
| `month`        |          1,969 | `int64` |
| `year`         |          1,969 | `int64` |
| `d`            |          1,969 | `str`   |
| `event_name_1` |            162 | `str`   |
| `event_type_1` |            162 | `str`   |
| `event_name_2` |              5 | `str`   |
| `event_type_2` |              5 | `str`   |
| `snap_CA`      |          1,969 | `int64` |
| `snap_TX`      |          1,969 | `int64` |
| `snap_WI`      |          1,969 | `int64` |

Initial memory usage was approximately **263.1 KB**.

Although the calendar dataset is relatively small, the columns were downcast where appropriate to establish a more efficient representation before writing the processed data to Parquet and to reduce memory consumption during subsequent processing.

### Integer Columns

The following integer columns were downcasted to smaller unsigned integer types:

| Column     | Before  | After    |
| ---------- | ------- | -------- |
| `wm_yr_wk` | `int64` | `uint16` |
| `wday`     | `int64` | `uint8`  |
| `month`    | `int64` | `uint8`  |
| `year`     | `int64` | `uint16` |
| `snap_CA`  | `int64` | `uint8`  |
| `snap_TX`  | `int64` | `uint8`  |
| `snap_WI`  | `int64` | `uint8`  |

Memory usage of the integer columns decreased from approximately **0.11 MB to 0.02 MB**, representing an **81% reduction**.

The conversion is appropriate because the values are non-negative and their ranges can be represented by smaller unsigned integer types.

### SNAP Columns

The `snap_CA`, `snap_TX`, and `snap_WI` columns are binary flags containing values of `0` or `1`. They indicate whether SNAP purchases were allowed in the respective state on a given date.

These columns are currently retained separately because they correspond to different states.

**Decision:** After joining the calendar data with the sales dataset, these state-specific SNAP indicators may be transformed into a more convenient representation if required for modelling or analysis.

### String/Object Columns

The following string columns were converted to pandas `category` dtype:

| Column         | Before | After      |
| -------------- | ------ | ---------- |
| `weekday`      | `str`  | `category` |
| `d`            | `str`  | `category` |
| `event_name_1` | `str`  | `category` |
| `event_type_1` | `str`  | `category` |
| `event_name_2` | `str`  | `category` |
| `event_type_2` | `str`  | `category` |

Memory usage decreased from approximately **0.12 MB to 0.04 MB**, representing a reduction of approximately **67%**.

Pandas categorical dtype stores unique category values separately and represents observations internally using integer codes. This can significantly reduce memory usage for columns containing repeated values.

The event-related columns also contain missing values. Pandas categorical dtype preserves these missing values separately from the category codes, so the original missing-value semantics are maintained.

### `d` Column

The `d` column contains identifiers such as:

```text
d_1
d_2
d_3
...
d_1913
```

Although this column has relatively high cardinality compared with columns such as `cat_id` or `state_id`, it was converted to `category` for the calendar dataset.

**Decision:** This is acceptable because the complete calendar dataset is processed as a single DataFrame. Therefore, pandas can construct one consistent category dictionary containing all `d` values.

This approach should **not automatically be applied to the chunked sales dataset**, where each chunk may contain only a subset of the possible categories.

### Overall Calendar Memory Reduction

| Metric             |  Before |     After |
| ------------------ | ------: | --------: |
| Total memory usage | 0.24 MB |   0.16 MB |
| Reduction          |       — | **33.3%** |

**Decision:** Downcasting and categorical conversion are retained for the calendar dataset because they reduce memory usage without changing the semantic meaning of the data.

---

## Prices Dataset

The M5 `sell_prices` dataset contains **4 columns** and has an initial memory usage of approximately **318.15 MB**.

The following conversions were applied:

* Integer columns → smaller unsigned integer types where the value range allows it.
* `float64` → `float32` where the precision is sufficient for price data.
* String/object columns → `category` where appropriate.

After conversion, memory usage decreased substantially:

| Metric             | Memory Usage |
| ------------------ | -----------: |
| Before downcasting |    318.15 MB |
| After downcasting  |     58.78 MB |
| Reduction          |     **~81%** |

**Decision:** Downcasting is retained for the prices dataset because the dataset is relatively large and the resulting reduction in memory usage is significant.

The reduced representation also lowers memory requirements for subsequent joins and transformations.

---

## Sales Dataset

The original sales dataset is stored in a wide format, with daily sales represented across columns such as:

```text
d_1, d_2, d_3, ..., d_1913
```

After melting the dataset into a long format, the resulting dataset contains approximately **58.3 million rows**.

The melted dataset requires approximately **7.37 GB** of memory when represented in pandas.

Because of its size, the sales dataset is processed **in chunks** rather than loaded and transformed as a single DataFrame.

### Chunk Processing Strategy

Numeric columns are downcast within each chunk where possible.

For example:

```python
chunk["sales"] = pd.to_numeric(
    chunk["sales"],
    downcast="integer"
)
```

This is safe because numeric downcasting selects a smaller dtype based on the values in the chunk while preserving the values themselves.

### String/Object Columns

String/object columns such as:

```text
item_id
dept_id
cat_id
store_id
state_id
d
```

are currently **not converted to pandas `category` dtype during chunk processing**.

The reason is that when pandas infers categories independently for each chunk, each chunk can have a different category dictionary and therefore a different internal integer-code mapping.

For example, one chunk might contain:

```text
cat_id
------
FOODS
```

while another chunk might contain:

```text
cat_id
------
HOBBIES
HOUSEHOLD
```

Pandas could therefore create mappings such as:

```text
Chunk 1:
0 → FOODS

Chunk 2:
0 → HOBBIES
1 → HOUSEHOLD
```

The original category labels are **not lost**. However, the integer codes are local to each chunk and should not be treated as globally consistent identifiers.

This can become problematic if the categorical columns are later combined or if the internal integer codes are interpreted as though the same code always represents the same category across all chunks.

### Decision

For the chunked sales ingestion process:

* Numeric columns are downcast within each chunk.
* String/object identifier columns remain as strings.
* Categorical conversion can be considered later after the complete dataset is consolidated.


This approach prioritises correctness and simplicity during the initial ingestion stage while still providing memory savings through numeric downcasting.

---

## Overall Decision

The data type optimisation strategy is dataset-specific:

| Dataset  | Numeric Downcasting | Category Conversion | Reason                                                   |
| -------- | ------------------- | ------------------- | -------------------------------------------------------- |
| Calendar | Yes                 | Yes                 | Small dataset; complete category vocabulary is available |
| Prices   | Yes                 | Yes                 | Large dataset; substantial memory reduction              |
| Sales    | Yes, per chunk      | No                  | Not requied for numeric data                             |

The primary objective is to **reduce memory and storage requirements without changing the semantic meaning of the source data**.

For the large sales dataset, correctness and predictable chunk processing take priority over applying categorical encoding prematurely.


