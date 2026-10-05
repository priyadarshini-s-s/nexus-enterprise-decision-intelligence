{{ config(materialized='view') }}

SELECT
    o.order_id,
    o.customer_id,
    c.customer_unique_id,
    o.order_status,

    -- Purchase date/time dimensions
    o.order_purchase_timestamp,
    CAST(o.order_purchase_timestamp AS DATE) AS purchase_date,
    EXTRACT(YEAR FROM o.order_purchase_timestamp) AS purchase_year,
    EXTRACT(MONTH FROM o.order_purchase_timestamp) AS purchase_month,
    EXTRACT(DAY FROM o.order_purchase_timestamp) AS purchase_day,
    EXTRACT(DOW FROM o.order_purchase_timestamp) AS purchase_day_of_week,
    EXTRACT(HOUR FROM o.order_purchase_timestamp) AS purchase_hour,

    -- Delivery timestamps
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,

    -- Existing delivery metric
    o.delivery_delay_days,
    o.is_late_delivery,

    -- Derived delivery duration
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
             AND o.order_purchase_timestamp IS NOT NULL
        THEN DATE_DIFF(
            'day',
            CAST(o.order_purchase_timestamp AS DATE),
            CAST(o.order_delivered_customer_date AS DATE)
        )
        ELSE NULL
    END AS delivery_duration_days,

    -- Delivery performance classification
    CASE
        WHEN o.order_status != 'delivered'
            THEN 'not_delivered'

        WHEN o.order_delivered_customer_date IS NULL
            THEN 'delivery_data_missing'

        WHEN o.is_late_delivery = 1
            THEN 'late'

        WHEN o.is_late_delivery = 0
            THEN 'on_time'

        ELSE 'unknown'
    END AS delivery_performance

FROM {{ ref('stg_olist_orders') }} o

LEFT JOIN {{ ref('stg_olist_customers') }} c
    ON o.customer_id = c.customer_id