{{ config(materialized='table') }}

SELECT
    purchase_date,

    COUNT(*) AS total_orders,

    COUNT(DISTINCT customer_id) AS unique_customers,

    SUM(
        CASE
            WHEN order_status = 'delivered'
            THEN 1
            ELSE 0
        END
    ) AS delivered_orders,

    SUM(
        CASE
            WHEN delivery_performance = 'late'
            THEN 1
            ELSE 0
        END
    ) AS late_orders,

    SUM(
        CASE
            WHEN delivery_performance = 'on_time'
            THEN 1
            ELSE 0
        END
    ) AS on_time_orders,

    SUM(
        CASE
            WHEN delivery_performance = 'not_delivered'
            THEN 1
            ELSE 0
        END
    ) AS not_delivered_orders,

    SUM(
        CASE
            WHEN delivery_performance = 'delivery_data_missing'
            THEN 1
            ELSE 0
        END
    ) AS delivery_data_missing_orders,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN delivery_performance = 'late'
                THEN 1
                ELSE 0
            END
        )
        / NULLIF(
            SUM(
                CASE
                    WHEN order_status = 'delivered'
                    THEN 1
                    ELSE 0
                END
            ),
            0
        ),
        2
    ) AS late_delivery_rate_pct,

    ROUND(
        AVG(
            CASE
                WHEN delivery_duration_days >= 0
                THEN delivery_duration_days
                ELSE NULL
            END
        ),
        2
    ) AS avg_delivery_duration_days

FROM {{ ref('int_olist_orders_enriched') }}

GROUP BY purchase_date
ORDER BY purchase_date