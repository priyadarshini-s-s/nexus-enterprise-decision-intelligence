{{ config(materialized='view') }}

SELECT
    product_category_name,
    product_category_name_english

FROM read_parquet(
    'D:/Projects/NEXUS/data/silver/olist/category_translation.parquet'
)