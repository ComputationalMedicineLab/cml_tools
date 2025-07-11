SELECT DISTINCT
    P.person_id,
    C.concept_id,
    B.condition_start_date AS datetime
FROM {source}.{omop}.person P
JOIN {source}.{omop}.condition_occurrence B ON (B.person_id = P.person_id)
JOIN {source}.{omop}.concept C ON (C.concept_id = B.condition_concept_id)
WHERE (
    P.birth_datetime >= '{birthdate}'
    AND B.condition_start_date >= '{startdate}'
    AND C.domain_id = 'Condition'
    AND C.vocabulary_id = 'SNOMED'
)
