{{ config(materialized='view') }}

SELECT
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date

FROM read_parquet(
    'D:/Projects/NEXUS/data/silver/olist/order_items.parquet'
)