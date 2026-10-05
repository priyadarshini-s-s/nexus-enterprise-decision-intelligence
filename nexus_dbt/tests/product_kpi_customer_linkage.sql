SELECT
    oi.order_id,
    oi.order_item_id

FROM {{ ref('stg_olist_order_items') }} oi

LEFT JOIN {{ ref('stg_olist_orders') }} o
    ON oi.order_id = o.order_id

LEFT JOIN {{ ref('stg_olist_customers') }} c
    ON o.customer_id = c.customer_id

WHERE c.customer_unique_id IS NULL