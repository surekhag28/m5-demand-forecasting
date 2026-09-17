# Feature Decisions

## Purpose

This document records the reasoning used to identify the candidate feature sets for the M5 demand forecasting model.

The feature selection process follows:

**EDA observation → Decision question → EDA conclusion → Candidate features**

At this stage, these are only **candidate features** based on the EDA findings. Their importance and usefulness will be tested later using time-based model experiments.

---

## Business Questions

The M5 demand forecasting project focuses on two main business questions:

1. **Sales likelihood** – When is an item-store combination likely to have a sale?
2. **Sales magnitude** – If a sale happens, how many units are likely to be sold?

---

## 1. Sales Likelihood / Intermittency

### Decision Question

Can recent sales frequency help the model predict whether an item-store combination will have positive demand?

### EDA Conclusion

The M5 dataset contains a large number of zero-sales observations, but the level of intermittency is different across item-store series.

Some series have frequent positive sales, while others have long periods of zero sales even after the product has been launched.

Because of this, the model should have information about the recent likelihood of a sale rather than relying only on sales quantity.

### Candidate Features

* `rolling_positive_rate_7`
* `rolling_positive_rate_28`
* `rolling_zero_rate_28`
* `days_since_last_sale`

---

## 2. Sales Demand Magnitude

### Decision Question

When an item-store combination has a sale, how much is likely to be sold?

### EDA Conclusion

Sales are highly right-skewed and contain a large number of zero-sales observations.

The sales magnitude also varies across different items and stores.

Therefore, the model should have features that describe the recent sales magnitude when sales actually occur.

### Candidate Features

* `rolling_positive_mean_7`
* `rolling_positive_mean_28`
* `rolling_positive_median_28`
* `rolling_positive_max_28`

---

## 3. Recent Demand

### Decision Question

Does recent historical sales demand provide useful information about future demand?

### EDA Conclusion

The time-series analysis showed that sales behaviour varies across item-store series.

Some series have frequent sales and stronger short-term patterns, while more intermittent series contain longer periods of zero sales.

Strong weekly seasonality and moderate monthly seasonality were also observed in some series. This means sales from the same weekday in previous weeks may provide useful information about future demand.

### Candidate Features

* `lag_1`
* `lag_7`
* `lag_14`
* `lag_28`

---

## 4. Recent Demand Level

### Decision Question

Can a smoothed representation of recent sales provide information about the current demand level for an item-store combination?

### EDA Conclusion

Daily M5 sales are noisy, especially for intermittent series. Individual daily sales can vary significantly across different series.

Rolling averages can reduce the effect of daily fluctuations and provide a more stable view of the recent demand level.

### Candidate Features

* `rolling_mean_7`
* `rolling_mean_28`
* `rolling_mean_56`

---

## 5. Product Lifecycle

### Decision Question

How should the model distinguish the period before a product was launched from the period when the product was actually available for sale?

### EDA Conclusion

Some products were launched later in the dataset. This means there are zero-sales observations before the product was available.

These zeros should not be treated as zero demand because the product was not available to be sold.

The model therefore needs information about product availability and how long the product has been available.

### Candidate Features

* `days_since_launch`
* `is_available`
* `launch_date`

---

## 6. Calendar and Seasonality

### Decision Question

Can calendar information explain recurring demand patterns that historical sales alone may not fully capture?

### EDA Conclusion

The analysis showed weekly and monthly seasonality, especially in series with more frequent sales.

Demand varies across different days of the week, and some mont

### Candidate Features

* `weekday`
* `week_of_year`
* `month`
* `year`
* `is_weekend`



Final Candidate Feature Table:

## Feature Catalogue

| Feature Group | Feature | Decision Question | EDA Conclusion | Purpose |
|---|---|---|---|---|
| Sales Likelihood / Intermittency | `rolling_positive_rate_7` | Can recent sales frequency help predict whether an item-store combination will have positive demand? | Intermittency varies across item-store series, with some series having frequent sales and others having long zero-sales periods after launch. | Recent likelihood of a sale |
| Sales Likelihood / Intermittency | `rolling_positive_rate_28` | Can recent sales frequency help predict whether an item-store combination will have positive demand? | Intermittency varies across item-store series, with some series having frequent sales and others having long zero-sales periods after launch. | Medium-term likelihood of a sale |
| Sales Likelihood / Intermittency | `rolling_zero_rate_28` | Can recent zero-sales frequency help predict whether an item-store combination will have positive demand? | Some item-store series contain a high proportion of zero-sales observations even after product launch. | Recent intermittency |
| Sales Likelihood / Intermittency | `days_since_last_sale` | Can the time since the previous sale indicate the likelihood of another sale? | Intermittent series can have long gaps between positive sales. | Recency of sales occurrence |
| Sales Demand Magnitude | `rolling_positive_mean_7` | When an item-store combination has a sale, how much is likely to be sold? | Positive sales are right-skewed and demand magnitude varies across item-store series. | Recent conditional demand magnitude |
| Sales Demand Magnitude | `rolling_positive_mean_28` | When an item-store combination has a sale, how much is likely to be sold? | Positive sales are right-skewed and demand magnitude varies across item-store series. | Medium-term conditional demand magnitude |
| Sales Demand Magnitude | `rolling_positive_median_28` | What is the typical quantity sold when a sale occurs? | Positive sales contain large values and are right-skewed, making the median less sensitive to extreme spikes. | Typical conditional demand |
| Sales Demand Magnitude | `rolling_positive_max_28` | Have there been recent unusually large sales? | Some item-store series experience occasional demand spikes. | Recent demand spike |
| Recent Demand | `lag_1` | Does the most recent sales observation provide information about future demand? | Sales behaviour varies across series and recent observations may contain short-term demand information. | Immediate demand signal |
| Recent Demand | `lag_7` | Does demand from the same weekday in the previous week provide useful information? | Strong weekly seasonality was observed in series with more frequent sales. | Weekly demand recurrence |
| Recent Demand | `lag_14` | Does demand from two weeks earlier provide useful information? | Weekly patterns can persist across multiple weeks. | Short-term weekly recurrence |
| Recent Demand | `lag_28` | Does demand from four weeks earlier provide useful information? | Monthly/longer weekly-cycle patterns were observed in some series. | Longer recurring demand signal |
| Recent Demand Level | `rolling_mean_7` | Can recent average sales provide information about the current demand level? | Daily sales are noisy, particularly for intermittent series. | Short-term demand level |
| Recent Demand Level | `rolling_mean_28` | Can a medium-term average provide a more stable demand signal? | Rolling averages can reduce daily fluctuations and represent underlying demand more smoothly. | Medium-term demand level |
| Recent Demand Level | `rolling_mean_56` | Can a longer-term average capture the underlying demand baseline? | Demand patterns vary across series and longer windows can provide a more stable baseline. | Long-term demand level |
| Product Lifecycle | `days_since_launch` | How should the model distinguish newly launched products from established products? | Some products have zero sales before they become available, which should not be interpreted as zero demand. | Product lifecycle stage |
| Product Lifecycle | `is_available` | Was the product actually available for sale on the given day? | Pre-launch zeros represent product unavailability rather than observed zero demand. | Availability status |
| Calendar / Seasonality | `day_of_week` | Does demand vary by day of the week? | Weekly seasonality was observed, particularly in less intermittent series. | Weekly seasonality |
| Calendar / Seasonality | `week_of_year` | Does demand vary across different periods of the year? | Recurring seasonal patterns were observed across the time series. | Annual seasonality |
| Calendar / Seasonality | `month` | Do some months consistently have different demand levels? | Monthly patterns were observed, with some months showing recurring differences across years. | Monthly seasonality |
| Calendar / Seasonality | `year` | Does demand behaviour change across years? | Demand levels and patterns can change over the multi-year observation period. | Long-term time variation |
| Calendar / Seasonality | `is_weekend` | Does demand differ between weekdays and weekends? | Demand behaviour differs across days of the week, including weekends. | Weekend effect |
| Calendar / Seasonality | `date` | Can the date provide useful time information to the model? | Date provides the underlying time index from which calendar and temporal features are derived. | Time reference |
| Product / Store Association | `item_id` | Do individual products exhibit different demand behaviour? | Demand patterns vary substantially across products. | Item-specific behaviour |
| Product / Store Association | `store_id` | Do different stores exhibit different demand behaviour? | Item-store series show different demand patterns across stores. | Store-specific behaviour |
| Product / Store Association | `state_id` | Does demand behaviour differ across states? | Store and regional characteristics can contribute to differences in demand behaviour. | Regional behaviour |
| Product / Store Association | `cat_id` | Does demand differ across product categories? | Different product categories show different levels of sales frequency and intermittency. | Category behaviour |
| Product / Store Association | `dept_id` | Does demand differ across departments? | Product departments contain different demand patterns across stores. | Department behaviour |
| Price | `sell_price` | Does the current selling price relate to demand? | Price varies across items and stores, and lower-priced products were observed to sometimes have higher sales. | Current price effect |
| Price | `price_change` | Does a change in price relate to demand? | Changes in selling price may affect customer demand. | Price change signal |
| Price | `price_change_pct` | Does the size of a price change relate to demand? | Relative price changes may provide more information than the absolute change alone. | Relative price effect |
| Price | `days_since_price_change` | Does the time since a price change relate to demand? | Demand response may differ depending on how recently the price changed. | Price-change recency |
| External Business Drivers | `snap` | Does SNAP affect sales likelihood or demand magnitude? | SNAP showed a slight increase in positive-sales rates and demand, with stronger differences observed for some product categories. | SNAP effect |
| External Business Drivers | `event_type` | Do events affect sales behaviour? | Event days showed different sales behaviour compared with non-event days. | Event category effect |
| External Business Drivers | `event_name` | Do specific events affect sales behaviour differently? | Different events may have different relationships with demand. | Snap and Event-specific effect |