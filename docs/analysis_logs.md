# M5 Demand Analysis

## Stage 1: Understanding the Sales Data

1. The grain of the sales data is **item × store × date**.

2. Each **item × store combination** represents an individual time series, with its own sales and demand pattern that we eventually need to forecast.

3. The sales distribution is highly right-skewed. Most observations have **zero or low sales**, while a small number of observations have very high sales.

4. The upper tail (high-volume sales) is mostly seen on **weekends** and for **FOOD and HOUSEHOLD** items.

5. Some high-volume sales also occur on weekdays, but these are often associated with factors such as **events, SNAP, or lower selling prices**.

6. Most of the very high-volume sales are associated with **lower selling prices**.

7. Around **68% of all observations have zero sales**, while around **31.8% have positive sales** across all stores and items.

8. Of the observations with zero sales, around **30% of item-store combinations have zero sales before the product launch**. The remaining zero sales are mostly occurring after launch, which represents intermittent demand rather than products that had not yet been launched.

9. Post-launch zero-sales days are very common and vary considerably across stores and items. The distribution is spread out, with many item-store combinations having around **75–85% zero-sales days after launch**.

10. A small number of item-store combinations have almost **99% zero-sales days**, but these combinations are relatively few.

11. This indicates that many item-store combinations have **highly intermittent demand**.

---

## Stage 2: Understanding Sales Over Time

1. From the yearly trend, we can observe that the **frequency of positive sales decreases from 2011 to around 2015**, followed by some improvement in 2016. This means that sales become more intermittent over much of the period.

2. When sales do occur, the average sales volume generally decreases from 2011 to 2015 and improves slightly in 2016. However, this does **not mean that overall product demand has necessarily declined**. The change could be related to factors such as new products being introduced, changes in prices, or changes in the mix of products being sold.

3. When we drill down into the **year and month level**, we see a downward trend in average sales from 2011 to around 2014/2015, followed by some improvement in 2016. Sales fluctuate from month to month, but there is **no strong and consistent monthly seasonal pattern** across the years.

4. Some months, such as **January, February, and March**, show relatively similar behaviour across many years, except for 2011. There is also some increase in sales during the middle of the year, but this pattern is not consistently repeated across all years.

5. The increase in sales in 2016 should not automatically be interpreted as seasonality or an increase in demand. It could be related to changes in prices, changes in the product mix, high-volume sales of certain products, events, or SNAP.

6. At the **weekly level**, we can observe a clearer seasonal pattern. Sales generally start increasing around **Friday, peak around Saturday/Sunday, and then decrease after Monday**.

7. At the category level, **FOOD products have higher average sales than HOBBIES and HOUSEHOLD products**. However, sales in all categories generally decline from 2012 to 2015 and then start to stabilise in 2016. We cannot conclude that demand for all products in a category declined because individual products and stores can have very different demand patterns.

8. Across the categories, the **frequency of positive sales is lowest for HOBBIES, followed by HOUSEHOLD**, and a similar pattern can be seen across stores. Overall, sales are intermittent across all stores.

9. Average sales are also heterogeneous across stores. For example, **CA_3, followed by TX_2 and WI_2**, have higher average FOOD sales than other stores. However, this does not necessarily mean that overall demand is higher in these stores. The difference could be driven by certain products selling in higher quantities, lower prices during certain weeks, state-specific events, or differences in customer preferences.

---

## Stage 3: Understanding Business and Contextual Drivers

### SNAP

1. At the aggregate level, when **SNAP is active**, average sales increase slightly and the percentage of zero-sales observations decreases slightly. This suggests some association between SNAP and sales. However, this alone does not show that SNAP caused the increase. Other factors, such as events, weekends, or lower prices, could also be involved.

2. When SNAP is not active, average sales are slightly higher on event days compared with regular days. When SNAP is active, sales are slightly higher regardless of whether it is an event day or not. However, we still cannot conclude that SNAP is directly driving sales because other factors may be contributing to the difference.

3. When SNAP is active, the percentage of zero-sales observations decreases slightly and average sales increase slightly on both weekdays and weekends. This suggests that there is **some association between SNAP and demand**, but it should not be interpreted as a causal effect.

4. When we analyse the data at the **item × store level**, the frequency of positive sales does not improve substantially when SNAP is active. Intermittent demand is still present for most combinations. However, some item-store combinations show slightly higher sales when SNAP is active.

5. The association appears to be somewhat stronger for **FOOD products**, which is reasonable given the nature of the SNAP programme. However, other factors such as price, weekday, events, and the specific products available in each store may also contribute to the observed pattern.

6. Overall, SNAP appears to have a **small association with sales**, but it does not appear to be the only factor explaining demand.

### Selling Price

1. When we analyse selling price, we can see that **lower-priced products generally have a higher frequency of positive sales and higher average sales** than products in higher price ranges.

2. However, this does not mean that expensive products necessarily have lower demand. Expensive products may naturally have lower purchase frequency because customers do not buy them as often.

3. Similarly, lower-priced products do not always have higher demand. A lower price may encourage customers to buy more units at once or stock up, which could result in higher sales during certain periods.

4. When we drill down to the **item level**, sales behaviour becomes more heterogeneous. Some items have low sales even when their prices are low, which may indicate naturally low demand or the influence of other factors such as weekday, events, or timing.

5. For some items, sales increase when the price increases compared with an earlier period. This does not mean that higher prices caused higher sales. It may simply indicate that the product had higher underlying demand during that particular period.

6. Very expensive products, particularly those with prices above **$100**, tend to have more intermittent sales and very low average sales. This may be because customers purchase these products less frequently or because their underlying demand is lower.

7. Overall, **selling price has some association with sales, but price alone does not strongly explain the sales pattern**. The relationship varies considerably across products and stores, so price needs to be considered together with other factors such as product, store, time, and promotions/events.