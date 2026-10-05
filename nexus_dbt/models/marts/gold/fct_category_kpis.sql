{{ config(materialized='table') }}

SELECT
    product_category_name,
    product_category_name_english,

    COUNT(*) AS product_count,

    SUM(order_item_count) AS order_item_count,

    SUM(unique_orders) AS product_order_count,

    SUM(unique_customers) AS product_customer_reach,

    SUM(unique_sellers) AS seller_count

FROM {{ ref('fct_product_kpis') }}

GROUP BY
    product_category_name,
    product_category_name_english