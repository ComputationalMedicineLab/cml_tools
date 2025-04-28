SELECT DISTINCT
    P.person_id,
    S.concept_id,
    M.measurement_datetime AS datetime,
    M.value_as_number
FROM {source}.{omop}.person P
JOIN {source}.{omop}.measurement M ON (M.person_id = P.person_id)
JOIN {workspace}.{schema}.measurement_stats S ON (S.concept_id = M.measurement_concept_id)
WHERE (
    P.birth_datetime >= '{birthdate}'
    AND M.measurement_datetime >= '{startdate}'
    AND M.value_as_number IS NOT NULL
    -- Outlier / error filtration; this is equiv. to the condition:
    -- "If p01 is positive, then the value must also be positive"
    AND (S.p01 <= 0 OR M.value_as_number > 0)
    AND (M.value_as_number < S.high_threshold)
    -- Remove specific non-lab LOINCs
    AND S.concept_code != '29463-7'     -- Body Weight
    AND S.concept_code != '8302-2'      -- Body Height
    AND S.concept_code != '39156-5'     -- Body Mass index
)
