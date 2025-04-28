-- This is a fragment to be composed by UNION in the CTE of select_meta.sql
SELECT DISTINCT
    concept_id,
    {fill_expr} AS fill_value,
    '{mode_str}' AS data_mode
FROM {workspace}.{schema}.{metatable}
