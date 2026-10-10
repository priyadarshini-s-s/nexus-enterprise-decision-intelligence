{{ config(materialized='view') }}

-- Portable dbt staging model using the canonical Silver Parquet source.
-- Explicit dbt lineage dependency:
-- depends_on: {{ source('olist', 'orders') }}

SELECT
    order_id,
    customer_id,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,

    CASE
        WHEN order_delivered_customer_date IS NOT NULL
         AND order_estimated_delivery_date IS NOT NULL
        THEN date_diff(
            'day',
            CAST(order_estimated_delivery_date AS DATE),
            CAST(order_delivered_customer_date AS DATE)
        )
        ELSE NULL
    END AS delivery_delay_days,

    CASE
        WHEN order_delivered_customer_date IS NULL
          OR order_estimated_delivery_date IS NULL
        THEN NULL
        WHEN CAST(order_delivered_customer_date AS DATE)
           > CAST(order_estimated_delivery_date AS DATE)
        THEN 1
        ELSE 0
    END AS is_late_delivery

FROM read_parquet(
    '{{ var("nexus_data_root") }}/silver/olist/orders.parquet'
)
