-- This is a fragment to be composed by UNION in the CTE of select_meta.sql
-- The metatable is often just the data itself itself; sometimes it is a
-- different table with aggregate information (e.g. measurement_stats.sql)
SELECT DISTINCT
    concept_id,
    {fill_expr} AS fill_value,
    {mode_expr} AS data_mode
FROM {workspace}.{schema}.{metatable}
