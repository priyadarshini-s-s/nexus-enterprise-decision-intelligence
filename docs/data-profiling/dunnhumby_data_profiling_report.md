# Dunnhumby Complete Journey - Silver Validation Report

## 1. Scope

This report records the initial Bronze-to-Silver transformation
and validation of the eight Dunnhumby Complete Journey source tables.

Raw CSV files remain unchanged. Silver tables are stored in
`data/silver/dunnhumby/` as Parquet files.

## 2. Source-to-Silver reconciliation

| Table | Source rows | Silver rows | Result |
|---|---:|---:|---|
| campaign_desc | 30 | 30 | PASS |
| campaign_table | 7,208 | 7,208 | PASS |
| causal_data | 36,786,524 | 36,786,524 | PASS |
| coupon | 124,548 | 124,548 | PASS |
| coupon_redempt | 2,318 | 2,318 | PASS |
| hh_demographic | 801 | 801 | PASS |
| product | 92,353 | 92,353 | PASS |
| transaction_data | 2,595,732 | 2,595,732 | PASS |
| **Total** | **39,609,514** | **39,609,514** | **PASS** |

The source and Silver column names and column order matched for all
eight tables in the reproducible validation.

## 3. Candidate-key checks

The following configured candidate keys passed null and duplicate-group
checks:

- `campaign_desc`: `CAMPAIGN`
- `campaign_table`: (`household_key`, `CAMPAIGN`)
- `coupon_redempt`: (`household_key`, `DAY`, `COUPON_UPC`, `CAMPAIGN`)
- `hh_demographic`: `household_key`
- `product`: `PRODUCT_ID`

These checks do not establish a unique natural key for `transaction_data`
or `causal_data`.

## 4. Referential integrity

All four configured checks passed with zero unmatched distinct keys:

- `campaign_table.CAMPAIGN` -> `campaign_desc.CAMPAIGN`
- `coupon.PRODUCT_ID` -> `product.PRODUCT_ID`
- `causal_data.PRODUCT_ID` -> `product.PRODUCT_ID`
- `transaction_data.PRODUCT_ID` -> `product.PRODUCT_ID`

The demographic table remains a partial household lookup: 801 of the
2,500 households observed in transactions have demographic records.

## 5. Known data characteristics

- The product catalog has 30,652 observed blank cells across
  `CURR_SIZE_OF_PRODUCT`, `DEPARTMENT`, `COMMODITY_DESC`, and
  `SUB_COMMODITY_DESC` (30,607 + 15 + 15 + 15).
- Coupon UPCs can map to multiple products and campaigns.
- Exact coupon mapping combinations can repeat in the source.
- Coupon redemption does not uniquely identify a purchased product.
- Large transaction quantities occur in gasoline and miscellaneous
  categories; quantities have not been globally capped.
- `display` and `mailer` values are preserved as categorical codes.
- Source-relative day/week fields have not been converted into calendar
  timestamps.
- The exact natural key of transaction lines remains unresolved.
- The candidate grain and key of `causal_data` remain unresolved.

## 6. Causal-analysis limitations

Successful ingestion and referential-integrity checks do not establish
causal identification.

Before estimating campaign effects, NEXUS must define and justify:

- treatment and treatment timing;
- eligible and exposed populations;
- outcome and outcome window;
- comparison-group construction;
- overlapping campaign handling;
- confounding and identification assumptions;
- robustness and sensitivity checks.

Campaign assignment, coupon mapping, redemption, and observed purchases
must remain distinct concepts.

## 7. Reproducibility

Transformation script:
`ingestion/transform_dunnhumby_silver.py`

Validation script:
`ingestion/validate_dunnhumby_silver.py`

Validation result: PASS for all checks implemented by the validator.

## 8. Outstanding items

- Confirm the dataset-specific license and redistribution terms.
- Validate the natural key/grain of `causal_data`.
- Investigate transaction-line key behavior.
- Confirm promotional code definitions from authoritative documentation.
- Design and validate the Gold analytical dataset before causal modelling.

This report records observed checks and known limitations; it does not
claim that every source-quality rule or causal assumption has been tested.


### causal_data grain investigation

DuckDB validation of the Silver Parquet table found 36,786,524 rows and
36,771,279 distinct (PRODUCT_ID, STORE_ID, WEEK_NO) tuples. There are
15,245 repeated groups, each with exactly two rows. No nulls were found
in the three candidate key columns.

Among the repeated groups, 15,208 have multiple display values and 105
have multiple mailer values; these counts may overlap. There are zero
excess exact duplicates across all five columns
(PRODUCT_ID, STORE_ID, WEEK_NO, display, mailer).

Decision: the three-column candidate key is not unique. The true source
grain remains unresolved pending authoritative definitions of display
and mailer. Preserve all source records; do not deduplicate or infer
treatment assignment from the categorical codes.


### documented display and mailer semantics

The Complete Journey promotions reference defines `display` as an in-store
placement code and `mailer` as a mailer/ad placement code. Code `0` means
no recorded placement; the other observed codes identify placement categories.
These are nominal categories, not numeric scales.

Reference:
- https://bradleyboehmke.github.io/completejourney/reference/promotions.html
- https://raw.githubusercontent.com/Lanbig/CSC465-visualization-project/master/Dataset/dunnhumby%20-%20The%20Complete%20Journey%20User%20Guide.pdf

The documented expected product-store-week grain conflicts with the observed
15,245 repeated candidate-key groups in this copy of the data. All repeated
groups have variation in `display` and/or `mailer`, and no exact duplicate
five-column rows were detected. Preserve all records and keep the actual
grain unresolved until this discrepancy is explained. Placement codes alone
do not establish causal effects.


### source artifact and observed grain clarification

The raw CSV and Silver Parquet both contain 36,786,524 rows. The raw CSV
independently reproduces 15,245 repeated PRODUCT_ID-STORE_ID-WEEK_NO
groups, so this finding is not introduced by the Silver transformation.

No duplicate full five-column combinations were found in the validation
query. Thus, the observed five-column combination is unique in this file,
but the intended business grain remains unresolved.

The Kaggle dataset inventory and an independent Complete Journey file
inventory both list 36,786,524 rows for causal_data.csv, matching this
raw CSV and its Silver Parquet output. The smaller processed R-package
artifact is not directly comparable evidence of a source-version mismatch.
No download or transformation defect has been demonstrated; the intended
business grain remains unresolved.

References:
- https://bradleyboehmke.github.io/completejourney/reference/promotions.html
- https://github.com/bltap-plmarket/dunnhumby-complete-journey/blob/main/docs/SCHEMA.md

Preserve the source rows. Do not silently deduplicate or aggregate them
until the source version and intended grain are understood.
