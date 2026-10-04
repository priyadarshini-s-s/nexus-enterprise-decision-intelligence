\# NEXUS — Demand Forecasting Decision Record



\## 1. Decision



The current NEXUS demand forecasting champion is a:



\*\*One-step-ahead Naive forecast\*\*



The forecast for day `t` is the observed order count from day `t-1`.



\\\[

\\hat{y\_t} = y\_{t-1}

\\]



This model is selected based on performance on the untouched chronological test period.



\---



\## 2. Business Question



NEXUS needs to estimate short-term daily order demand to support operational decision-making such as:



\- capacity planning

\- fulfillment planning

\- staffing

\- operational workload estimation

\- demand monitoring



The forecasting target is deliberately defined using a quantity that is directly observable in the Olist dataset.



\### Target



\*\*Daily unique order demand\*\*



For each calendar day:



`order\_count = count(unique order\_id)`



The timestamp used is:



`order\_purchase\_timestamp`



The project does not invent product quantities because the Olist

`order\_items` table does not provide a conventional quantity field.



\---



\## 3. Source Data



Primary source:



\*\*Olist Brazilian E-Commerce Public Dataset\*\*



The source contains approximately 100,000 real anonymized e-commerce orders.



Relevant source table:



`orders`



Relevant fields:



\- `order\_id`

\- `order\_purchase\_timestamp`



The source data was transformed through the NEXUS Bronze → Silver → Gold architecture.



\---



\## 4. Demand Series Construction



The complete observed source period is:



`2016-09-04 → 2018-10-17`



The source series was converted to a continuous daily calendar.



Days without observed orders were retained as zero-demand days.



Complete source series:



\- 774 calendar days

\- 99,441 orders

\- 140 zero-demand days

\- mean: 128.48 orders/day

\- median: 124 orders/day

\- maximum: 1,176 orders/day



\---



\## 5. Modeling Window



The modeling window was restricted to:



`2017-01-01 → 2018-08-31`



Reason:



The earliest portion of the source series is sparse, while the later

portion after August 2018 also contains very low observed activity.



The complete source series remains preserved.



The modeling artifact is a derived analytical view and does not overwrite

the source representation.



\---



\## 6. Temporal Split



Random splitting was deliberately avoided because forecasting must respect

time.



\### Training



`2017-01-01 → 2018-03-31`



455 days



\### Validation



`2018-04-01 → 2018-06-30`



91 days



\### Test



`2018-07-01 → 2018-08-31`



62 days



The test period was kept untouched during model selection.



\---



\## 7. Candidate Models



The following approaches were evaluated.



\### Baselines



1\. Naive

2\. Seasonal Naive

3\. 7-day Moving Average



\### Classical Time-Series Models



4\. Additive Holt-Winters

5\. Damped Holt-Winters

6\. Seasonal-only Exponential Smoothing

7\. Trend-only Exponential Smoothing



\### Trend-Aware Baselines



8\. Drift

9\. Damped Drift

10\. Seasonal Naive + Drift

11\. Weekday-adjusted Naive



\### Machine Learning



12\. XGBoost regression



XGBoost experiments included:



\- feature-set ablation

\- tree-depth comparison

\- minimum-child-weight comparison

\- early stopping

\- final retraining on train + validation



\---



\## 8. Feature Engineering



The XGBoost model used 26 features.



\### Calendar features



\- day\_of\_week

\- day\_of\_month

\- month

\- quarter

\- week\_of\_year

\- is\_weekend



\### Trend



\- time\_index



\### Lag features



\- lag\_1

\- lag\_2

\- lag\_3

\- lag\_7

\- lag\_14

\- lag\_21

\- lag\_28



\### Rolling features



7-day:



\- rolling\_mean\_7

\- rolling\_std\_7

\- rolling\_min\_7

\- rolling\_max\_7



14-day:



\- rolling\_mean\_14

\- rolling\_std\_14

\- rolling\_min\_14

\- rolling\_max\_14



28-day:



\- rolling\_mean\_28

\- rolling\_std\_28

\- rolling\_min\_28

\- rolling\_max\_28



All rolling features were calculated using prior observations only to

avoid target leakage.



\---



\## 9. Validation Results



Selected validation results:



| Model | Validation MAE | Validation RMSE |

|---|---:|---:|

| Naive | 38.6374 | 50.5706 |

| Seasonal Naive | 38.2967 | 54.6840 |

| Moving Average | 45.0832 | 52.6556 |

| Damped Holt-Winters | 44.8959 | 57.9802 |

| Additive Holt-Winters | 45.2460 | 57.7048 |

| Drift | 58.0492 | 73.0635 |

| Weekday-adjusted Naive | 60.8856 | 73.5314 |

| XGBoost with early stopping | 30.7844 | 39.1079 |



XGBoost produced the strongest validation result.



However, validation performance alone was not used to select the final

model.



\---



\## 10. Final Test Evaluation



After model selection, the test period was evaluated once.



Evaluation method:



\*\*One-step-ahead forecasting\*\*



For each test day:



`prediction(t) = actual(t-1)`



Final results:



| Model | Test MAE | Test RMSE |

|---|---:|---:|

| \*\*Naive\*\* | \*\*35.6290\*\* | \*\*44.7652\*\* |

| XGBoost | 38.9056 | 49.9591 |

| Seasonal Naive | 69.8226 | 85.3814 |



\---



\## 11. Final Decision



\### Selected model



\*\*Naive one-step-ahead forecasting\*\*



Test MAE:



`35.6290`



Test RMSE:



`44.7652`



XGBoost was:



`9.20%`



worse than Naive on test MAE.



Seasonal Naive was:



`95.97%`



worse than Naive on test MAE.



\---



\## 12. Why the Simpler Model Was Selected



The Naive model was selected because it provided the strongest

generalization among the tested candidates on the untouched test period.



The important observation is that XGBoost performed substantially better

during validation but failed to maintain that advantage on the later test

period.



This indicates that relationships learned during the earlier period did

not generalize sufficiently to the later demand regime.



The result demonstrates why model complexity should not be treated as a

proxy for model quality.



NEXUS therefore follows:



\*\*Evidence → validation → untouched temporal test → model selection\*\*



rather than:



\*\*More complex model → automatically better model\*\*



\---



\## 13. Interpretation of Seasonality



The dataset shows a clear day-of-week pattern.



Average daily orders were approximately:



\- Monday: 145.91

\- Tuesday: 143.81

\- Wednesday: 140.11

\- Thursday: 134.19

\- Friday: 128.38

\- Saturday: 98.97

\- Sunday: 107.75



However, the Seasonal Naive model did not generalize well.



Therefore:



\*\*weekly seasonality exists in the data, but simple weekly repetition was

not sufficiently stable to produce the best forecasts in the final test

period.\*\*



\---



\## 14. Limitations



This model should not be interpreted as a universal demand forecasting

solution.



Important limitations include:



1\. The Olist dataset represents one historical e-commerce business context.

2\. The source period is historical rather than live.

3\. The target is order count, not physical product-unit demand.

4\. No external promotions, holidays, advertising spend, inventory,

&#x20;  competitor activity, or macroeconomic variables were incorporated.

5\. The source contains unusual demand spikes that may represent real

&#x20;  business events.

6\. The test period covers only 62 days.

7\. The model is predictive rather than causal.



\---



\## 15. Future Improvements



Potential future work includes:



\- external calendar/event features

\- promotion information where legitimately available

\- hierarchical product/category forecasting

\- seller-level demand forecasting

\- probabilistic prediction intervals

\- longer rolling-origin evaluation

\- additional legitimate high-volume time-series datasets

\- real-time demand monitoring when a legitimate live event source exists



Future improvements should only be adopted when supported by evaluation

evidence.



\---



\## 16. Artifacts



Important forecasting artifacts:



\- `data/gold/olist/daily\_order\_demand.parquet`

\- `data/gold/olist/daily\_order\_demand\_modeling.parquet`

\- `data/gold/olist/forecasting\_splits/`

\- `data/gold/olist/forecasting\_feature\_splits/`

\- `data/gold/olist/forecasting\_models/`

\- `data/gold/olist/exponential\_smoothing\_results.parquet`

\- `data/gold/olist/trend\_baseline\_results.parquet`

\- `data/gold/olist/xgboost\_feature\_ablation\_results.parquet`

\- `data/gold/olist/xgboost\_hyperparameter\_results.parquet`

\- `data/gold/olist/xgboost\_early\_stopping\_results.parquet`

\- `data/gold/olist/final\_forecasting\_champion\_results.parquet`

\- `data/gold/olist/final\_forecasting\_champion\_predictions.parquet`



\---



\## 17. Decision Status



\*\*Status: Accepted for the current NEXUS Olist forecasting vertical\*\*



The Naive model is the current benchmark/champion for this specific

forecasting task and evaluation setup.



It should not be described as a universally optimal forecasting model.

