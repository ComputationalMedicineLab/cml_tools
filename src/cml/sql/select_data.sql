-- Select observational datapoints merged across different subtables. The parts
-- are given by ./select_data_cte.sql and merged with UNION to make the CTE
WITH datapoints AS (
{cte}
)
SELECT DISTINCT
    person_id,
    concept_id,
    {date_expr} AS datetime,
    value::DOUBLE
FROM datapoints
JOIN {workspace}.{schema}.{concept_table} USING (concept_id)
JOIN {workspace}.{schema}.{cohort_table} USING (person_id)
ORDER BY person_id, concept_id, datetime
    ;
