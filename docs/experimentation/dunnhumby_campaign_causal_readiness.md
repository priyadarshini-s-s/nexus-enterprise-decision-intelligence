# Dunnhumby Campaign Causal-Readiness Assessment

## Status

Decision: NO-GO for a causal-effect or uplift claim using the current
campaign comparison design. Continue with data validation and explicitly
labelled observational analysis.

This is a design decision based on the diagnostics recorded below, not a
claim that causal inference is impossible with every potential design.

## Dataset and treatment interpretation

- Dataset: Dunnhumby The Complete Journey.
- Campaign assignment source: `campaign_table.csv`.
- Campaign timing source: `campaign_desc.csv`.
- Household outcomes: `transaction_data.csv`.
- Campaign assignment is observed, but randomization is not established.
- Campaign assignment must not be treated as proof of coupon receipt,
  exposure, or redemption.
- The product/store/week `causal_data.csv` table does not provide a
  household-level randomized treatment assignment.

## Baseline balance findings

Absolute standardized mean differences (SMDs) from the 28-day
pre-campaign comparison, after excluding households assigned to another
campaign whose date window overlaps the target campaign:

| Campaign | Type | Eligible assigned | Eligible comparison | Sales SMD | Basket SMD | Active-day SMD |
|---|---|---:|---:|---:|---:|---:|
| 8 | TypeA | 768 | 1359 | 0.8402 | 0.8032 | 0.9500 |
| 13 | TypeA | 636 | 1374 | 0.8632 | 0.8555 | 1.0254 |
| 18 | TypeA | 510 | 1251 | 0.6636 | 0.7635 | 0.9049 |
| 26 | TypeA | 310 | 2165 | 0.4844 | 0.4438 | 0.5418 |
| 29 | TypeB | 79 | 2057 | 0.4473 | 0.3732 | 0.4927 |
| 30 | TypeA | 272 | 2040 | 0.4867 | 0.3882 | 0.4643 |

Campaign 14 had a lower maximum SMD of 0.3312, but only five eligible
assigned households. That sample is too small to support a credible
standalone campaign-effect estimate.

The diagnostics show substantial differences in observed shopping
behaviour. They do not establish the full assignment mechanism or rule
out unmeasured confounding.

## Pre-campaign trends

For Campaigns 26, 29 and 30, assigned households had higher mean weekly
sales than comparison households in each of the eight pre-campaign weeks.
The aggregate trajectories do not establish a credible parallel-trends
assumption.

## Demographic coverage

For the eligible Campaign 30 population:

| Group | Eligible households | With demographics | Coverage |
|---|---:|---:|---:|
| Assigned | 272 | 130 | 47.79% |
| Comparison | 2040 | 573 | 28.09% |

Baseline shopping behaviour also differed between households with and
without demographic records. A demographics-complete-case analysis could
therefore change the study population and introduce further selection.

## Campaign 30 descriptive before/during comparison

The campaign ran from day 323 through day 369 (47 days). The comparison
used the preceding 47 days, days 276 through 322.

| Metric | Assigned mean change | Comparison mean change |
|---|---:|---:|
| Sales per household | -6.8366 | -0.1129 |
| Baskets per household | -0.2426 | +0.0500 |
| Active days per household | -0.1949 | +0.0368 |

The raw difference in sales changes is -6.7237 sales units per household.
This is a descriptive contrast only. It must not be presented as the
campaign's causal effect or used as a validated uplift target.

## Current decision

Do not publish an ATE, treatment-effect ranking, or uplift targeting
recommendation from these comparisons.

Before reconsidering causal estimation, establish:
1. The intended treatment and the meaning of household campaign assignment.
2. Whether actual exposure or coupon receipt can be identified.
3. The outcome definition and the correct observation window.
4. The assignment mechanism and plausible confounders.
5. Whether adequate covariate overlap and a defensible identification
   strategy exist.
6. Data-use/licensing requirements for the intended project use.

If these requirements cannot be met, retain the work as a transparent
observational campaign analysis and report its limitations. Do not invent
randomization, household exposure, joins, or causal evidence.

## Reproducibility

The local raw Dunnhumby CSV files are analyzed using DuckDB. The
reproducible diagnostic entry point is:

`python ingestion/analyze_dunnhumby_campaign_readiness.py`

The script calculates baseline household outcomes, absolute standardized
mean differences (SMDs), pre-campaign weekly sales, Campaign 30 descriptive
before/during changes, and demographic coverage. It does not fit a causal
or uplift model.

### Calculation conventions

- Baseline outcomes cover the 28 days immediately before each campaign
  start day.
- Household-level sales are summed over the window. Baskets and active
  days are counted as distinct `BASKET_ID` and `DAY` values.
- Eligible assigned households exclude households assigned to another
  campaign whose calendar window overlaps the target campaign.
- Comparison households are drawn from households observed in transaction
  data, exclude assignment to the target campaign, and exclude assignments
  overlapping the target campaign window. This is an eligibility screen,
  not proof of no exposure.
- The script calculates absolute SMD as the absolute difference in group
  means divided by the square root of the average of the two groups'
  sample variances. This convention reproduces the six reference SMD rows
  recorded above.
- Weekly sales comparisons use the eight complete `WEEK_NO` intervals
  immediately preceding the campaign-overlapping week. The selected weeks
  are 25-32 for Campaign 26, 33-40 for Campaign 29, and 39-46 for
  Campaign 30. Households with no transactions in a selected week have
  zero sales for that week.
- Campaign 30's before/during windows each cover 47 days: days 276-322
  and 323-369, respectively.

### Verification and limitations

The diagnostic run reproduced all six recorded baseline SMD rows to four
decimal places, the Campaign 30 descriptive before/during values, and
demographic coverage. The weekly diagnostics show higher mean weekly sales
for assigned households in every selected pre-campaign week for Campaigns
26, 29, and 30.

These results are descriptive and observational. Matching the reference
figures verifies computational reproducibility, not randomization,
actual campaign exposure, absence of confounding, or a causal identification
strategy. Do not present these contrasts as causal effects or validated
uplift targets.

Raw data remain excluded from Git. The source-data license remains
unverified.
