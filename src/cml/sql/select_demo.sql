-- Get, for each person in the core cohort, (a) birthdate, (b) sex, (c) race
-- We include both the source value and concept id; the source values can vary
-- slightly across implementations; its better to process in Py than SQL.
SELECT DISTINCT
    person_id,
    birth_datetime::DATE AS birth_date,
    gender_concept_id,
    GC.concept_name AS gender_concept_name,
    COALESCE(gender_source_concept_id, 2000003109) AS gender_source_concept_id,
    gender_source_value,
    race_concept_id,
    RC.concept_name AS race_concept_name,
    COALESCE(race_source_concept_id, 2003603112) AS race_source_concept_id,
    race_source_value
FROM {workspace}.{schema}.{cohort_table}
JOIN {source}.{omop}.person P USING (person_id)
JOIN {source}.{omop}.concept GC ON (P.gender_concept_id = GC.concept_id)
JOIN {source}.{omop}.concept RC ON (P.race_concept_id = RC.concept_id)
    ;
