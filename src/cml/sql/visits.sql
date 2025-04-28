-- Notes:
-- (1) 9201 is the OMOP concept for Inpatient Admission
-- (2) We require the visit have a duration (end > start); we could also cap
--     that duration to some interval (e.g. INTERVAL 3 MONTH, see below) and
--     set a floor (must be at least 30 minutes maybe?).
-- (3) This will not resolve visit intervals; it is possible that the results
--     will contain overlapping visits, or multiple visits beginning at the
--     same datetime.
SELECT DISTINCT
    V.person_id,
    V.visit_start_datetime AS start_datetime,
    V.visit_end_datetime AS end_datetime
FROM {source}.{omop}.visit_occurrence V
JOIN {source}.{omop}.person P USING (person_id)
WHERE (
    P.birth_datetime >= '{birthdate}'
    AND V.visit_start_datetime >= '{startdate}'
    AND V.visit_start_datetime < V.visit_end_datetime
    -- AND V.visit_end_datetime >= (V.visit_start_date + INTERVAL 1 MINUTE)
    -- AND V.visit_end_datetime <= (V.visit_start_date + INTERVAL 3 MONTH)
    AND V.visit_concept_id = 9201
)
