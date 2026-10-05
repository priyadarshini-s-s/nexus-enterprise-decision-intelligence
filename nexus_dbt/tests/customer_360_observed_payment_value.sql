SELECT
    customer_unique_id,
    payment_observation_status,
    total_payment_value

FROM {{ ref('dim_customer_360') }}

WHERE payment_observation_status = 'observed'
  AND total_payment_value IS NULL