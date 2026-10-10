{{ config(materialized='view') }}

SELECT
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date

FROM read_parquet(
    '{{ var("nexus_data_root") }}/silver/olist/order_items.parquet'
)