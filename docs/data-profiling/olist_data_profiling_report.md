# NEXUS — Olist Data Profiling Report

## 1. Dataset Overview

Source:
Olist Brazilian E-Commerce Public Dataset

Purpose:
Primary transactional business spine for NEXUS.

The dataset contains multiple related tables representing customers,
orders, order items, payments, reviews, products, sellers, geolocation,
and product-category translation.

---

## 2. Source Tables

| Table | Rows | Columns | Business Grain |
|---|---:|---:|---|
| customers | 99,441 | 5 | Customer record |
| geolocation | 1,000,163 | 5 | Geolocation observation |
| order_items | 112,650 | 7 | Order line item |
| order_payments | 103,886 | 5 | Payment record |
| order_reviews | 99,224 | 7 | Review record |
| orders | 99,441 | 8 | Order |
| products | 32,951 | 9 | Product |
| sellers | 3,095 | 4 | Seller |
| category_translation | 71 | 2 | Category mapping |

---

## 3. Identity and Grain Findings

### Customers

`customer_id` identifies the customer record associated with an order.

`customer_unique_id` represents persistent customer identity and is
therefore the appropriate identifier for longitudinal customer analysis.

The dataset contains:

- 99,441 customer records
- 96,096 unique persistent customers

Therefore:

> Customer record count must not be interpreted as unique customer count.

---

### Orders

The orders table contains 99,441 orders.

`order_id` is unique within the orders table.

Orders contain lifecycle timestamps including:

- purchase
- approval
- carrier handoff
- customer delivery
- estimated delivery

Missing lifecycle timestamps must be interpreted according to order
status and process applicability rather than blindly imputed.

---

### Order Items

The order-items table contains 112,650 rows and 98,666 unique orders.

This confirms that order-items operate at line-item grain.

The natural row identity should be treated as:

`(order_id, order_item_id)`

rather than assuming `order_item_id` is globally unique.

---

### Payments

The payments table contains 103,886 records and 99,440 unique orders.

Multiple payment records can therefore exist for a single order.

Order-level revenue/payment analysis must aggregate payment records
appropriately to avoid double counting.

---

### Reviews

The reviews table contains 99,224 rows.

Review text is sparse:

- `review_comment_title`: 88.34% null
- `review_comment_message`: 58.70% null

Therefore, structured review scores provide broader analytical
coverage than textual NLP.

NLP pipelines should explicitly handle missing review text.

---

## 4. Data Quality Findings

### Customers

- No null values detected.
- No duplicate rows detected.
- `customer_id` is unique.
- `customer_unique_id` contains repeated customer identities across
  customer records.

### Geolocation

- 1,000,163 rows.
- 261,831 duplicate rows.
- No null values detected.

The duplicate records should not be blindly removed before confirming
the intended grain and usage of the geolocation source.

### Products

Missingness is concentrated in:

- product category
- product name length
- product description length
- product photos quantity

There are only two missing values in each physical-dimension field
such as weight and dimensions.

### Sellers

- No null values detected.
- No duplicate rows detected.
- `seller_id` is unique.

### Category Translation

- 71 rows.
- No null values.
- Both source and translated category values are unique.

---

## 5. Referential Integrity

All tested foreign-key relationships produced zero orphan values.

| Child Relationship | Orphans | Status |
|---|---:|---|
| orders.customer_id → customers.customer_id | 0 | PASS |
| order_items.order_id → orders.order_id | 0 | PASS |
| order_items.product_id → products.product_id | 0 | PASS |
| order_items.seller_id → sellers.seller_id | 0 | PASS |
| order_payments.order_id → orders.order_id | 0 | PASS |
| order_reviews.order_id → orders.order_id | 0 | PASS |

This establishes structural consistency between the major Olist
transactional tables.

---

## 6. Important Analytical Risks

### Risk 1 — Double counting order value

Payment and order-item tables can contain multiple records per order.

Order-level metrics must establish a clear aggregation grain.

### Risk 2 — Confusing customer records with customers

`customer_id` and `customer_unique_id` have different analytical
meanings.

### Risk 3 — Treating missing timestamps as ordinary missing data

Missing delivery lifecycle timestamps may reflect order status or
process state.

### Risk 4 — Blind geolocation deduplication

Duplicate geolocation observations should not automatically be deleted.

### Risk 5 — Sparse review text

NLP coverage will be substantially lower than review-score coverage.

---

## 7. Referential Integrity Conclusion

The tested Olist relationships contain zero orphan foreign-key values.

The dataset is therefore structurally suitable for construction of
the NEXUS Silver layer, subject to the grain, missingness, and
business-rule considerations documented above.

---

## 8. Silver Layer Design Principles

The Silver layer will:

1. Preserve source meaning.
2. Standardize column names.
3. Apply explicit data types.
4. Normalize timestamps.
5. Preserve source keys.
6. Validate relationships.
7. Handle missing values according to business meaning.
8. Avoid premature aggregation.
9. Record data-quality metrics.
10. Maintain reproducibility.

Silver transformations must not silently alter business meaning.