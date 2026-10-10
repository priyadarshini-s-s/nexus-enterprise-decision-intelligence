{{ config(materialized='view') }}

SELECT
    product_id,
    product_category_name,
    product_name_lenght,
    product_description_lenght,
    product_photos_qty

FROM read_parquet(
    '{{ var("nexus_data_root") }}/silver/olist/products.parquet'
)