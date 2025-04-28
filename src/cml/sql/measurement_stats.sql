SELECT
    C.concept_id,
    C.concept_class_id,
    C.vocabulary_id,
    C.concept_code,
    C.concept_name,
    count(*) AS count,
    min(M.value_as_number) AS min_value,
    max(M.value_as_number) AS max_value,
    std(M.value_as_number) AS std_value,
    percentile_cont(0.0001) WITHIN GROUP (ORDER BY M.value_as_number) AS p0001,
    percentile_cont(0.001)  WITHIN GROUP (ORDER BY M.value_as_number) AS p001,
    percentile_cont(0.01)   WITHIN GROUP (ORDER BY M.value_as_number) AS p01,
    percentile_cont(0.05)   WITHIN GROUP (ORDER BY M.value_as_number) AS p05,
    percentile_cont(0.25)   WITHIN GROUP (ORDER BY M.value_as_number) AS p25,
    percentile_cont(0.5)    WITHIN GROUP (ORDER BY M.value_as_number) AS p50,
    percentile_cont(0.75)   WITHIN GROUP (ORDER BY M.value_as_number) AS p75,
    percentile_cont(0.95)   WITHIN GROUP (ORDER BY M.value_as_number) AS p95,
    percentile_cont(0.99)   WITHIN GROUP (ORDER BY M.value_as_number) AS p99,
    percentile_cont(0.999)  WITHIN GROUP (ORDER BY M.value_as_number) AS p999,
    percentile_cont(0.9999) WITHIN GROUP (ORDER BY M.value_as_number) AS p9999,
    p99 + abs(p99 - p50) AS high_threshold,
    p01 - abs(p01 - p50) AS low_threshold,
    p75 - p25 AS iqr
FROM {source}.{omop}.measurement M
JOIN {source}.{omop}.concept C ON (C.concept_id = M.measurement_concept_id)
-- Add EGFR (concept_id = '44806420'), which is classified as a procedure
WHERE (
    C.concept_class_id = 'Lab Test'
    OR C.concept_class_id = 'Clinical Observation'
    OR C.concept_id = '44806420'
)
GROUP BY
    C.concept_id,
    C.concept_class_id,
    C.vocabulary_id,
    C.concept_code,
    C.concept_name
HAVING p50 IS NOT NULL AND p25 != p75
ORDER BY count DESC
