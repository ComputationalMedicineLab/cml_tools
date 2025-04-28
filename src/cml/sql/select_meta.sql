-- The CTE comes by composition over select_meta_cte.sql; we do this because
-- differet projects may use different sets of underlying source tables, and
-- because the underlying sources may specify different ways to get mode/fill.
-- Everything else should be a simple function of the concept table.
WITH extra_meta AS (
{cte}
)
SELECT
    C.concept_id,
    C.concept_name,
    C.domain_id,
    C.vocabulary_id,
    C.concept_class_id,
    C.concept_code,
    M.data_mode,
    M.fill_value::DOUBLE
FROM {workspace}.{schema}.{concept_table}
JOIN {source}.{omop}.concept C USING (concept_id)
JOIN extra_meta M USING (concept_id)
ORDER BY M.data_mode, C.concept_id
    ;
