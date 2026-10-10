{{ config(materialized='view') }}

SELECT
    *
FROM read_parquet(
    '{{ var("nexus_data_root") }}/gold/olist/customer_360.parquet'
)