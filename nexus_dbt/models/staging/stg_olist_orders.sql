{{ config(materialized='view') }}

SELECT
    order_id,
    customer_id,
    order_status,
    order_purchase_timestamp,
    order_approved_at,
    order_delivered_carrier_date,
    order_delivered_customer_date,
    order_estimated_delivery_date,
    delivery_delay_days,
    is_late_delivery

FROM {{ iceberg_source(
    'D:/Projects/NEXUS/data/iceberg/olist/orders'
) }}