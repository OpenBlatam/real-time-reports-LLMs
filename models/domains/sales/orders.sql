-- orders.sql
WITH order_source AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

order_items AS (
    SELECT * FROM {{ ref('stg_order_items') }}
),

-- Join or additional transformation CTEs

final AS (
    SELECT
        id,                    -- Primary key
        user_id,               -- Relationship to users entity
        created_at,            -- Event timestamp
        status,                -- State attributes

        -- Include both dimensional and measure fields
        items_count,           -- Measure
        order_total,           -- Measure
        tax_amount,            -- Measure
        shipping_amount,       -- Measure

        payment_method,        -- Dimension
        channel,               -- Dimension
        is_first_purchase      -- Calculated dimension
    FROM order_source
    LEFT JOIN order_items USING (id)
    -- Additional joins as needed
)

SELECT * FROM final
