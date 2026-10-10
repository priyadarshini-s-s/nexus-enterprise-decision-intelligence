{{ config(materialized='view') }}

SELECT
    customer_id,
    customer_unique_id,
    customer_zip_code_prefix,
    customer_city,
    customer_state

FROM read_parquet(
    '{{ var("nexus_data_root") }}/silver/olist/customers.parquet'
)