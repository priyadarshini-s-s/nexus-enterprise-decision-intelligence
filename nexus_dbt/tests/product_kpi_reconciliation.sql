WITH source_metrics AS (

    SELECT
        COUNT(*) AS source_order_items,
        COUNT(DISTINCT product_id) AS source_products
    FROM read_parquet(
        'D:/Projects/NEXUS/data/silver/olist/order_items.parquet'
    )

),

product_metrics AS (

    SELECT
        SUM(order_item_count) AS product_kpi_order_items,
        COUNT(*) AS product_kpi_products
    FROM {{ ref('fct_product_kpis') }}

)

SELECT
    *
FROM source_metrics
CROSS JOIN product_metrics

WHERE source_order_items != product_kpi_order_items
   OR source_products != product_kpi_products