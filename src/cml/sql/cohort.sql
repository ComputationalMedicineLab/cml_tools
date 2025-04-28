-- Selects only persons from data table {tablename}, restricted to a given
-- concept index {concept_table}, who have at least one concept with two
-- distinct data points.
SELECT DISTINCT person_id
FROM {workspace}.{schema}.{name}
JOIN {workspace}.{schema}.{concept_table} USING (concept_id)
GROUP BY person_id, concept_id
HAVING count(DISTINCT datetime) >= 2
