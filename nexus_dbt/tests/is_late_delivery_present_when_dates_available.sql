SELECT *
FROM {{ ref('stg_olist_orders') }}
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
  AND is_late_delivery IS NULL
