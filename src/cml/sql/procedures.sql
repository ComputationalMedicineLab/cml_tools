SELECT DISTINCT
    PR.person_id,
    PR.procedure_concept_id AS concept_id,
    PR.procedure_datetime AS datetime
FROM {source}.{omop}.person P
JOIN {source}.{omop}.procedure_occurrence PR USING (person_id)
JOIN {source}.{omop}.concept C ON (C.concept_id = PR.procedure_concept_id)
WHERE (
    P.birth_datetime >= '{birthdate}'
    AND PR.procedure_datetime >= '{startdate}'
    AND C.domain_id = 'Procedure'
    AND C.vocabulary_id = 'CPT4'
)
