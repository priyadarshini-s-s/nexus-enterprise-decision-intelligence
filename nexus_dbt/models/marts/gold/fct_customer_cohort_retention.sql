{{ config(materialized='table') }}

WITH customer_orders AS (

    SELECT
        customer_unique_id,
        purchase_date
    FROM {{ ref('int_olist_orders_enriched') }}
    WHERE customer_unique_id IS NOT NULL
      AND purchase_date IS NOT NULL

),

customer_first_purchase AS (

    SELECT
        customer_unique_id,
        DATE_TRUNC('month', MIN(purchase_date)) AS cohort_month
    FROM customer_orders
    GROUP BY customer_unique_id

),

customer_activity AS (

    SELECT DISTINCT
        o.customer_unique_id,
        c.cohort_month,
        DATE_TRUNC('month', o.purchase_date) AS activity_month
    FROM customer_orders o
    INNER JOIN customer_first_purchase c
        ON o.customer_unique_id = c.customer_unique_id

),

cohort_activity AS (

    SELECT
        cohort_month,
        activity_month,

        DATE_DIFF(
            'month',
            cohort_month,
            activity_month
        ) AS retention_month,

        COUNT(DISTINCT customer_unique_id) AS active_customers

    FROM customer_activity
    GROUP BY
        cohort_month,
        activity_month

),

cohort_sizes AS (

    SELECT
        cohort_month,
        COUNT(DISTINCT customer_unique_id) AS cohort_size
    FROM customer_first_purchase
    GROUP BY cohort_month

)

SELECT
    a.cohort_month,
    a.activity_month,
    a.retention_month,
    s.cohort_size,
    a.active_customers,

    ROUND(
        100.0 * a.active_customers / NULLIF(s.cohort_size, 0),
        2
    ) AS retention_rate_pct

FROM cohort_activity a

INNER JOIN cohort_sizes s
    ON a.cohort_month = s.cohort_month

ORDER BY
    a.cohort_month,
    a.retention_month