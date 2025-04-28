SELECT DISTINCT
    P.person_id,
    C.concept_id,
    D.drug_exposure_start_datetime AS datetime
FROM {source}.{omop}.person P
JOIN {source}.{omop}.drug_exposure D ON (D.person_id = P.person_id)
JOIN {source}.{omop}.x_drug_exposure X ON (X.drug_exposure_id = D.drug_exposure_id)
JOIN {source}.{omop}.concept_ancestor CA  ON (CA.descendant_concept_id = D.drug_concept_id)
JOIN {source}.{omop}.concept C ON (C.concept_id = CA.ancestor_concept_id)
WHERE (
    P.birth_datetime >= '{birthdate}'
    AND D.drug_exposure_start_datetime >= '{startdate}'
    AND C.concept_class_id = 'Ingredient'
    AND (nullif(X.x_strength, '') IS NOT NULL OR nullif(X.x_dose, '') IS NOT NULL)
    AND nullif(X.x_frequency, '') IS NOT NULL
    AND D.route_source_value IS NOT NULL
    AND X.x_doc_stype = 'Problem list'
)
