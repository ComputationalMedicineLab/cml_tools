-- Requires a table with at least fields "concept_id", "person_id", and
-- "datetime". Generates counts per concept_id of unique persons, total
-- mentions, and total mentions per unique patient-date pair, as if the
-- underlying table had been generated at "date" resolution instead of
-- "timestamp".
SELECT DISTINCT
    concept_id,
    count(*) AS mentions,
    count(DISTINCT person_id, datetime::DATE) AS mentions_daily,
    count(DISTINCT person_id) AS persons
FROM {workspace}.{schema}.{tablename}
GROUP BY concept_id
