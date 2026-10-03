\# NEXUS Retention Intelligence — Experiment Record v1



\## 1. Problem Definition



NEXUS needs to identify customers who are likely to make another purchase within a future 90-day period.



The objective is not to classify permanent customer churn.



The prediction target is:



`repeat\_purchase\_90d`



A customer receives target:



\- `1` if the customer makes another purchase in `(T, T + 90 days]`

\- `0` otherwise



where `T` is the historical observation cutoff.



This formulation avoids treating a 90-day period without purchase as permanent churn.



\---



\## 2. Data Source



Primary business dataset:



Olist Brazilian E-Commerce Public Dataset.



The Olist data contains approximately 100K orders from the 2016–2018 observation period.



NEXUS uses the Olist transactional domain as its primary business spine.



The retention model is built from the validated Olist Silver layer and point-in-time Gold retention training table.



\---



\## 3. Prediction Grain



One row represents:



`customer × observation\_snapshot`



The same customer can therefore appear in multiple temporal snapshots.



The target only uses information after the observation cutoff.



\---



\## 4. Prediction Horizon



Prediction horizon:



`90 days`



Reason:



The Olist observation period provides substantially more temporal coverage for a 90-day horizon than longer horizons.



The 90-day horizon also provides a practical balance between:



\- sufficient future observation

\- usable number of training snapshots

\- meaningful retention decision window



\---



\## 5. Temporal Coverage



Monthly retention snapshots:



2016-12-31 through 2018-06-30



Total snapshots:



`19`



The final three months form the test period:



\- 2018-04

\- 2018-05

\- 2018-06



\---



\## 6. Temporal Split



\### TRAIN



December 2016 through December 2017



Rows:



`223,898`



Positive observations:



`1,888`



Positive rate:



`0.843241%`



\### VALIDATION



January 2018 through March 2018



Rows:



`172,334`



Positive observations:



`1,192`



Positive rate:



`0.6917%`



\### TEST



April 2018 through June 2018



Rows:



`232,263`



Positive observations:



`1,186`



Positive rate:



`0.5106%`



The split is temporal rather than random to reduce future-information leakage.



\---



\# 7. Feature Representation — V1



The production candidate uses 22 point-in-time features:



\- orders\_to\_date

\- total\_spend\_to\_date

\- total\_order\_items\_to\_date

\- unique\_products\_to\_date

\- unique\_categories\_to\_date

\- unique\_sellers\_to\_date

\- recency\_days

\- average\_order\_value

\- average\_items\_per\_order

\- orders\_30d

\- spend\_30d

\- items\_30d

\- orders\_60d

\- spend\_60d

\- items\_60d

\- orders\_90d

\- spend\_90d

\- items\_90d

\- average\_review\_score\_to\_date

\- payment\_observation\_available

\- review\_observation\_available

\- item\_observation\_available



All features are calculated using information available at or before the observation cutoff.



\---



\# 8. Feature Semantics



`total\_spend\_to\_date`



Sum of observed Olist `payment\_value` up to the cutoff.



It should not automatically be interpreted as recognized accounting revenue.



`total\_order\_items\_to\_date`



Count of Olist order-item records.



Olist does not provide a conventional quantity field in `order\_items`, so this represents observed line-item records rather than physical unit quantity.



`recency\_days`



Days between the customer's latest observed purchase and the observation cutoff.



`average\_order\_value`



Observed payment value divided by observed orders.



`\*\_30d`, `\*\_60d`, `\*\_90d`



Rolling behavioral measures calculated using only information available by the snapshot.



\---



\# 9. Models Evaluated



\## Model A — Balanced Logistic Regression



Configuration:



\- StandardScaler

\- LogisticRegression

\- class\_weight="balanced"

\- solver="lbfgs"

\- max\_iter=2000

\- random\_state=42



Test results:



ROC-AUC:



`0.612363`



PR-AUC:



`0.014753`



Brier score:



`0.20511799`



The model produced strong extreme-tail ranking performance but poor probability calibration.



\---



\## Model B — Unweighted Logistic Regression



Configuration:



\- StandardScaler

\- LogisticRegression

\- class\_weight=None

\- solver="lbfgs"

\- max\_iter=2000

\- random\_state=42



Test results:



ROC-AUC:



`0.615269`



PR-AUC:



`0.015453`



Brier score:



`0.00508950`



Mean absolute calibration gap:



`0.227112%`



This is the primary retention model candidate.



\---



\## Model C — Reduced V2 Logistic Regression



A reduced feature representation was evaluated to test whether feature simplification improved generalization.



Test results:



ROC-AUC:



`0.609905`



PR-AUC:



`0.014713`



Top-1% lift:



`6.579x`



The reduced representation did not improve test discrimination over V1.



\---



\## Model D — XGBoost Benchmark



Tested configuration:



\- binary logistic objective

\- 300 estimators

\- max\_depth=4

\- learning\_rate=0.05

\- min\_child\_weight=5

\- subsample=0.8

\- colsample\_bytree=0.8

\- reg\_lambda=1

\- scale\_pos\_weight based on training imbalance

\- histogram tree method

\- random\_state=42



Test results:



ROC-AUC:



`0.572090`



PR-AUC:



`0.011812`



The tested XGBoost configuration did not outperform the tested Logistic Regression candidates.



This does not establish that Logistic Regression is universally superior to XGBoost.



It establishes only the result of the configurations evaluated in this experiment.



\---



\# 10. Ranking Evaluation



Because the positive class is highly imbalanced, ranking metrics are important.



For the balanced V1 Logistic Regression on the test set:



| Capacity | Lift |

|---|---:|

| Top 1% | 7.000x |

| Top 2% | 5.059x |

| Top 5% | 3.238x |

| Top 10% | 2.496x |

| Top 20% | 1.792x |

| Top 30% | 1.481x |

| Top 50% | 1.266x |



The model therefore has practical value as a customer-ranking system.



The model should not be interpreted as establishing that an intervention causes additional purchases.



\---



\# 11. Probability Calibration



The balanced model had poor probability calibration.



Its class weighting substantially changed the probability scale.



The unweighted model showed much better calibration:



Brier score:



`0.00508950`



Mean absolute calibration gap:



`0.227112%`



Therefore the unweighted model is preferred as the canonical propensity model when predicted probabilities are consumed downstream.



\---



\# 12. Monthly Stability



The unweighted model was evaluated independently across the three monthly test populations.



| Month | Rows | Positive Rate | ROC-AUC | PR-AUC | Brier | Calibration Gap |

|---|---:|---:|---:|---:|---:|---:|

| 2018-04 | 70,954 | 0.627167% | 0.619748 | 0.019238 | 0.00622362 | 0.123382% |

| 2018-05 | 77,680 | 0.570288% | 0.605824 | 0.019053 | 0.00566300 | 0.168652% |

| 2018-06 | 83,629 | 0.356336% | 0.611759 | 0.010946 | 0.00359458 | 0.369420% |



ROC-AUC remained within:



`0.605824 – 0.619748`



across the three months.



PR-AUC was lower in June alongside substantially lower positive prevalence.



Calibration gaps remained below:



`0.4%`



in all three monthly populations.



\---



\# 13. Primary Model Contract



Model:



`Unweighted Logistic Regression V1`



Artifact:



`data/gold/olist/retention\_models/logistic\_retention\_unweighted.joblib`



Input:



22 point-in-time customer behavioral features.



Output:



`P(repeat\_purchase\_90d = 1)`



Primary uses:



1\. Retention propensity estimation

2\. Customer ranking

3\. Capacity-based intervention selection

4\. Decision Engine input



\---



\# 14. Why Logistic Regression?



The model is intentionally retained despite testing a more complex tree-based model.



Reasons supported by the experiments:



\- competitive discrimination in the tested configurations

\- substantially better calibration than the balanced alternative

\- simple inference

\- interpretable coefficients

\- low computational cost

\- easy deployment

\- stable monthly ROC-AUC across the tested test period

\- transparent relationship between model inputs and propensity score



The choice is evidence-based for this experiment rather than a claim that Logistic Regression is universally superior.



\---



\# 15. Known Limitations



\### 15.1 Observational data



The model predicts naturally occurring repeat purchases.



It does not estimate causal treatment effects.



\---



\### 15.2 Target definition



The target represents a future purchase event.



It does not establish that the purchase was completed successfully under a stricter business-status definition.



\---



\### 15.3 Temporal coverage



The source data ends in 2018.



Therefore model performance should not automatically be assumed to represent present-day ecommerce behavior.



\---



\### 15.4 Dataset concentration



The primary model is based on the Olist ecosystem.



External validation on another real ecommerce dataset is required before making broader generalization claims.



\---



\### 15.5 Feature redundancy



Several behavioral features are correlated.



This can make individual Logistic Regression coefficients difficult to interpret independently.



Model-level predictive performance is therefore emphasized over simplistic coefficient-based causal explanations.



\---



\### 15.6 No intervention effect



A high propensity score does not mean that contacting the customer will cause a purchase.



Causal/uplift modeling is a separate NEXUS module.



\---



\# 16. Experiment Decision



The current evidence supports freezing:



`Unweighted Logistic Regression V1`



as the canonical retention model candidate for NEXUS.



The balanced Logistic Regression, reduced V2 model, and XGBoost benchmark remain part of the experiment history.



No further hyperparameter tuning is performed at this stage.



The next stage is operationalization:



1\. MLflow experiment tracking

2\. Model artifact logging

3\. Model registry

4\. Reproducibility metadata

5\. Serving contract

6\. FastAPI integration

7\. Decision Engine integration

8\. Monitoring

