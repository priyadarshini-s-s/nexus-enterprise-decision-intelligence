{{ config(materialized='view') }}

SELECT
    product_category_name,
    product_category_name_english

FROM read_parquet(
    '{{ var("nexus_data_root") }}/silver/olist/category_translation.parquet'
)