SELECT DISTINCT person_id, datetime
FROM {workspace}.{schema}.{name}
JOIN {workspace}.{schema}.{concept_table} USING (concept_id)
JOIN {workspace}.{schema}.{cohort_table} USING (person_id)
