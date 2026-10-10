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

The figures in this report are based on read-only DuckDB diagnostics run
against the local raw Dunnhumby CSV files. The analysis scripts and
environment should be versioned separately when finalized. Raw data remain
excluded from Git.
