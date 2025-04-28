SELECT
    person_id,
    start_datetime::TIMESTAMP_NTZ AS start_datetime,
    end_datetime::TIMESTAMP_NTZ AS end_datetime
FROM {workspace}.{schema}.visits
JOIN {workspace}.{schema}.{cohort_table} USING (person_id)
ORDER BY person_id, start_datetime, end_datetime
    ;
