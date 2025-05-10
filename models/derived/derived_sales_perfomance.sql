-- derived_sales_performance.sql
WITH user_orders AS (
    SELECT
        user_id,
        SUM(order_total) as total_spent,
        COUNT(id) as order_count,
        MIN(created_at) as first_order_date,
        MAX(created_at) as last_order_date
    FROM {{ ref('orders') }} -- Reference to entity model
    WHERE status = 'complete' -- Status depends on your data
    GROUP BY 1
),

user_attributes AS (
    SELECT
        id as user_id,
        segment,
        created_at as joined_date
    FROM {{ ref('users') }} -- Reference to entity model
),

final AS (
    SELECT
        user_attributes.user_id,
        segment,
        joined_date,

        -- Entity fields from orders
        total_spent,
        order_count,
        first_order_date,
        last_order_date,

        -- Derived calculations
        total_spent / NULLIF(order_count, 0) as average_order_value,
        DATEDIFF('day', first_order_date, last_order_date) as customer_tenure_days,
        DATEDIFF('day', last_order_date, CURRENT_DATE()) as days_since_last_order,

        -- Complex derived metrics
        CASE
            WHEN days_since_last_order <= 30 THEN 'active'
            WHEN days_since_last_order <= 90 THEN 'at_risk'
            ELSE 'inactive'
        END as activity_status
    FROM user_attributes
    LEFT JOIN user_orders USING (user_id)
)

SELECT * FROM final
