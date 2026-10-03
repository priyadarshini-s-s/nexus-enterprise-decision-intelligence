# NEXUS Canonical Data Model

## 1. Purpose

The canonical model defines the business meaning and valid relationships of NEXUS data without pretending that independent public datasets share customer or product identities.

## 2. Domains

| Domain | Source | Primary purpose |
|---|---|---|
| Transactional Business | Olist | Customer, order, product, seller, payment, delivery, review intelligence |
| Marketing / Causal | dunnhumby Complete Journey | Household transactions, campaigns, coupons, response analysis |
| Digital Journey | RetailRocket | Visitor behavior, views, carts, transactions |
| Large-Scale Recommendation | OTTO | Session sequences, clicks, carts, orders |

## 3. Canonical entity register

| Entity | Source domain | Grain | Primary identity | Main time field |
|---|---|---|---|---|
| Customer | Olist | Customer/order-address record | `customer_id`; persistent identity via `customer_unique_id` | — |
| Order | Olist | One order | `order_id` | purchase timestamp |
| OrderItem | Olist | One product line within an order | (`order_id`, `order_item_id`) | shipping limit timestamp |
| Product | Olist | One product | `product_id` | — |
| Seller | Olist | One seller | `seller_id` | — |
| Payment | Olist | One payment record for an order | order/payment relationship | — |
| Review | Olist | One review associated with an order | review/order relationship | review creation timestamp |
| Household | dunnhumby | One household | dataset-local household ID | — |
| Campaign | dunnhumby | One campaign | campaign ID | campaign dates |
| Coupon | dunnhumby | One coupon/product/customer relationship | source-defined key | campaign/coupon timing |
| Redemption | dunnhumby | One coupon redemption event | source-defined key | redemption timing |
| Visitor | RetailRocket | One anonymous visitor | `visitor_id` | — |
| Event | RetailRocket | One visitor-item interaction | event record | event timestamp |
| Session | OTTO | One anonymous behavioral session | `session` | event timestamp |
| Recommendation Event | OTTO | One event inside a session | session + event sequence | event timestamp |

## 4. Valid Olist relationships

```text
Customer.customer_id -> Order.customer_id
Order.order_id -> OrderItem.order_id
Order.order_id -> Payment.order_id
Order.order_id -> Review.order_id
OrderItem.product_id -> Product.product_id
OrderItem.seller_id -> Seller.seller_id
```

For customer-level longitudinal analysis, `customer_unique_id` is the persistent customer identity exposed by Olist's schema.

## 5. Invalid cross-domain joins

These are NOT valid without an explicit source mapping:

```text
Olist.customer_unique_id  != dunnhumby.household_id
Olist.customer_unique_id  != RetailRocket.visitor_id
Olist.customer_unique_id  != OTTO.session
Olist.product_id          != RetailRocket.item_id
Olist.product_id          != OTTO.aid
```

A matching data type, similar name, or overlapping numeric value is not evidence of identity.

## 6. Derived NEXUS entities

Derived entities are computed from observed source data:

- `customer_360`
- `customer_rfm`
- `customer_cohorts`
- `customer_features`
- `product_360`
- `product_features`
- `daily_demand`
- `seller_performance`
- `review_sentiment`
- `delivery_performance`
- `campaign_response`
- `journey_funnel`
- `session_features`
- `recommendation_candidates`
- `recommendation_scores`
- `anomaly_events`
- `decisions`

## 7. Layering

```text
SOURCE
  |
  v
BRONZE
  Exact source representation
  |
  v
SILVER
  Typed + cleaned + validated + standardized
  |
  v
GOLD
  Business-ready analytical products
  |
  +--> BI
  +--> ML
  +--> APIs
  +--> Decision Engine
```

## 8. Time semantics

NEXUS must distinguish:

- `event_time`: when the business event happened in the source
- `ingestion_time`: when NEXUS received/loaded the record
- `processing_time`: when a transformation or streaming job processed it

These timestamps must not be silently substituted for one another.

## 9. Identity principle

NEXUS is a unified decision-intelligence platform, not a fabricated unified customer table.

Independent datasets remain domain-isolated unless an authoritative crosswalk exists.
