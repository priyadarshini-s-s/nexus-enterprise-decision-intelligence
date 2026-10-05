WITH kpi AS (

    SELECT *
    FROM {{ ref('fct_executive_kpis') }}

)

SELECT
    *
FROM kpi
WHERE
    delivered_orders != (
        late_orders
        + on_time_orders
        + delivery_data_missing_orders
    )

    OR ROUND(
        repeat_customer_rate_pct,
        2
    ) != ROUND(
        100.0 * repeat_customers / NULLIF(total_customers, 0),
        2
    )

    OR ROUND(
        overall_late_delivery_rate_pct,
        2
    ) != ROUND(
        100.0 * late_orders / NULLIF(delivered_orders, 0),
        2
    )