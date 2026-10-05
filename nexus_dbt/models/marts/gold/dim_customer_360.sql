{{ config(materialized='view') }}

SELECT
    *
FROM read_parquet(
    'D:/Projects/NEXUS/data/gold/olist/customer_360.parquet'
)