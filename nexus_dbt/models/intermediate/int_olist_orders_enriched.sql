{{ config(materialized='view') }}

SELECT
    order_id,
    customer_id,
    order_status,

    -- Purchase date/time dimensions
    order_purchase_timestamp,
    CAST(order_purchase_timestamp AS DATE) AS purchase_date,
    EXTRACT(YEAR FROM order_purchase_timestamp) AS purchase_year,
    EXTRACT(MONTH FROM order_purchase_timestamp) AS purchase_month,
    EXTRACT(DAY FROM order_purchase_timestamp) AS purchase_day,
    EXTRACT(DOW FROM order_purchase_timestamp) AS purchase_day_of_week,
    EXTRACT(HOUR FROM order_purchase_timestamp) AS purchase_hour,

    -- Delivery timestamps
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,

    -- Existing delivery metric
    delivery_delay_days,
    is_late_delivery,

    -- Derived delivery duration
    CASE
        WHEN order_delivered_customer_date IS NOT NULL
             AND order_purchase_timestamp IS NOT NULL
        THEN DATE_DIFF(
            'day',
            CAST(order_purchase_timestamp AS DATE),
            CAST(order_delivered_customer_date AS DATE)
        )
        ELSE NULL
    END AS delivery_duration_days,

    -- Delivery performance classification
    CASE
        WHEN order_status != 'delivered'
            THEN 'not_delivered'

        WHEN order_delivered_customer_date IS NULL
            THEN 'delivery_data_missing'

        WHEN is_late_delivery = 1
            THEN 'late'

        WHEN is_late_delivery = 0
            THEN 'on_time'

        ELSE 'unknown'
    END AS delivery_performance

FROM {{ ref('stg_olist_orders') }}