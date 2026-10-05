{{ config(materialized='table') }}

WITH orders AS (

    SELECT *
    FROM {{ ref('fct_olist_order_performance') }}

),

customers AS (

    SELECT *
    FROM {{ ref('dim_customer_360') }}

),

order_metrics AS (

    SELECT
        SUM(total_orders) AS total_orders,
        SUM(delivered_orders) AS delivered_orders,
        SUM(late_orders) AS late_orders,
        SUM(on_time_orders) AS on_time_orders,
        SUM(not_delivered_orders) AS not_delivered_orders,
        SUM(delivery_data_missing_orders) AS delivery_data_missing_orders,

        AVG(late_delivery_rate_pct) AS avg_daily_late_delivery_rate_pct,
        AVG(avg_delivery_duration_days) AS avg_delivery_duration_days

    FROM orders

),

customer_metrics AS (

    SELECT
        COUNT(*) AS total_customers,

        SUM(
            CASE
                WHEN is_repeat_customer THEN 1
                ELSE 0
            END
        ) AS repeat_customers,

        SUM(total_payment_value) AS total_payment_value,

        AVG(
            CASE
                WHEN total_orders > 0
                THEN total_payment_value / total_orders
            END
        ) AS average_customer_order_value

    FROM customers

)

SELECT

    -- Volume
    o.total_orders,
    o.delivered_orders,

    -- Customers
    c.total_customers,
    c.repeat_customers,

    ROUND(
        100.0 * c.repeat_customers / NULLIF(c.total_customers, 0),
        2
    ) AS repeat_customer_rate_pct,

    -- Delivery
    o.late_orders,
    o.on_time_orders,
    o.not_delivered_orders,
    o.delivery_data_missing_orders,

    ROUND(
        100.0 * o.late_orders / NULLIF(o.delivered_orders, 0),
        2
    ) AS overall_late_delivery_rate_pct,

    ROUND(
        100.0 * o.on_time_orders / NULLIF(o.delivered_orders, 0),
        2
    ) AS overall_on_time_delivery_rate_pct,

    -- Customer value
    c.total_payment_value,

    ROUND(
        c.total_payment_value / NULLIF(o.total_orders, 0),
        2
    ) AS overall_average_order_value,

    ROUND(
        c.average_customer_order_value,
        2
    ) AS average_customer_order_value,

    -- Supporting operational metrics
    ROUND(
        o.avg_delivery_duration_days,
        2
    ) AS avg_delivery_duration_days

FROM order_metrics o
CROSS JOIN customer_metrics c