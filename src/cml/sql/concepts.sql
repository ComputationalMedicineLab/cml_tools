SELECT DISTINCT concept_id
FROM {workspace}.{schema}.conditions
GROUP BY concept_id
HAVING count(*) >= 1000
    UNION
SELECT DISTINCT concept_id
FROM {workspace}.{schema}.medications
GROUP BY concept_id
HAVING count(*) >= 1000
    UNION
SELECT DISTINCT concept_id
FROM {workspace}.{schema}.measurements
GROUP BY concept_id
HAVING count(*) >= 1000 AND count(DISTINCT person_id) >= 10
