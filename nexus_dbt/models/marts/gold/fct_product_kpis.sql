{{ config(materialized='table') }}

WITH item_base AS (

    SELECT
        oi.order_id,
        oi.order_item_id,
        oi.product_id,
        oi.seller_id
    FROM {{ ref('stg_olist_order_items') }} oi

),

order_customers AS (

    SELECT
        o.order_id,
        c.customer_unique_id
    FROM {{ ref('stg_olist_orders') }} o
    LEFT JOIN {{ ref('stg_olist_customers') }} c
        ON o.customer_id = c.customer_id

),

product_attributes AS (

    SELECT
        p.product_id,
        p.product_category_name,
        t.product_category_name_english,
        p.product_name_lenght,
        p.product_description_lenght,
        p.product_photos_qty

    FROM {{ ref('stg_olist_products') }} p

    LEFT JOIN {{ ref('stg_olist_category_translation') }} t
        ON p.product_category_name = t.product_category_name

),

product_metrics AS (

    SELECT
        i.product_id,

        COUNT(*) AS order_item_count,

        COUNT(DISTINCT i.order_id) AS unique_orders,

        COUNT(DISTINCT oc.customer_unique_id) AS unique_customers,

        COUNT(DISTINCT i.seller_id) AS unique_sellers

    FROM item_base i

    LEFT JOIN order_customers oc
        ON i.order_id = oc.order_id

    GROUP BY
        i.product_id

)

SELECT
    m.product_id,

    a.product_category_name,
    a.product_category_name_english,

    a.product_name_lenght,
    a.product_description_lenght,
    a.product_photos_qty,

    m.order_item_count,
    m.unique_orders,
    m.unique_customers,
    m.unique_sellers

FROM product_metrics m

LEFT JOIN product_attributes a
    ON m.product_id = a.product_id