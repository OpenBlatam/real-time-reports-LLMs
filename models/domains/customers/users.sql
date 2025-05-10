-- users.sql
WITH source AS (
    SELECT * FROM {{ ref('stg_users') }}
),

-- Additional transformation/enrichment CTEs as needed

final AS (
    SELECT
        id,                    -- Primary key
        first_name,            -- Identity attributes
        last_name,
        email,
        segment,               -- Categorization attributes
        created_at,            -- Temporal attributes
        updated_at,

        -- Include both dimensional and measure fields directly on the entity
        lifetime_value,        -- Measure: calculated or aggregated values
        is_active,             -- Dimension: status flags
        last_login_date        -- Dimension: activity timestamps

        -- All attributes related to this entity in one model
    FROM source
    LEFT JOIN {{ ref('int_user_activity') }} USING (id)  -- Join intermediate calculations if needed
)

SELECT * FROM final
