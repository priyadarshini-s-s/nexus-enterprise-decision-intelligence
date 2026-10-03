\# NEXUS Retention Model Serving Contract



\## Model



Registered model:



`logistic\_retention`



Serving candidate:



`Version 2`



Model type:



Unweighted Logistic Regression



Feature version:



`V1`



\---



\## Prediction Target



`repeat\_purchase\_90d`



Definition:



Probability that a customer with observed purchase history makes another purchase within the following 90 days.



\---



\## Input Contract



The model requires exactly 22 features:



1\. orders\_to\_date

2\. total\_spend\_to\_date

3\. total\_order\_items\_to\_date

4\. unique\_products\_to\_date

5\. unique\_categories\_to\_date

6\. unique\_sellers\_to\_date

7\. recency\_days

8\. average\_order\_value

9\. average\_items\_per\_order

10\. orders\_30d

11\. spend\_30d

12\. items\_30d

13\. orders\_60d

14\. spend\_60d

15\. items\_60d

16\. orders\_90d

17\. spend\_90d

18\. items\_90d

19\. average\_review\_score\_to\_date

20\. payment\_observation\_available

21\. review\_observation\_available

22\. item\_observation\_available



\---



\## Input Rules



All 22 fields are required.



No target column may be supplied during inference.



All feature values must correspond to information available at the prediction cutoff.



Features must not contain future information.



Missing values are not accepted by the current serving contract.



\---



\## Output Contract



The service returns:



`repeat\_purchase\_probability`



Range:



`\[0, 1]`



Interpretation:



Estimated probability of a repeat purchase within 90 days under the observational prediction definition.



\---



\## Business Interpretation



The score is a predictive propensity.



It is NOT:



\- a causal treatment effect

\- guaranteed incremental revenue

\- guaranteed customer response to an intervention

\- permanent churn probability



A high score means the model estimates a higher likelihood of naturally occurring repeat purchase.



\---



\## Decision Engine Usage



The prediction may be used for:



\- customer ranking

\- capacity-based targeting

\- retention prioritization

\- downstream decision analysis



The model should not automatically determine intervention eligibility without business rules.



The Decision Engine should separately account for:



\- available intervention capacity

\- intervention cost

\- business constraints

\- causal/uplift evidence where available



\---



\## Model Lineage



Registered model:



`logistic\_retention`



Version:



`2`



Feature version:



`V1`



Training strategy:



Temporal train/validation/test split



Prediction horizon:



90 days



Source experiment:



`NEXUS-Retention-Intelligence`



\---



\## Limitations



The model is trained on historical Olist ecommerce data.



The model's historical performance should not automatically be interpreted as present-day ecommerce performance.



The model predicts naturally occurring repeat purchase and does not estimate intervention effects.



External validation is required before broad generalization.

