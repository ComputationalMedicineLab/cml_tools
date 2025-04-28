-- See ./select_data.sql
SELECT DISTINCT person_id, concept_id, datetime, {value_expr}
FROM {workspace}.{schema}.{name}
