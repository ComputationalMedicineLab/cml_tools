SELECT DISTINCT
    C.concept_id,
    C.concept_name,
    C.domain_id,
    C.vocabulary_id,
    C.concept_class_id,
    C.concept_code,
    CASE
        WHEN C.vocabulary_id = 'VUMC Gender' THEN 'Sex'
        WHEN C.vocabulary_id = 'VUMC Race' THEN 'Race'
    END AS data_mode,
    0.0::DOUBLE AS fill_value
FROM {source}.{omop}.concept C
WHERE C.vocabulary_id IN ('VUMC Gender', 'VUMC Race')
    ;
