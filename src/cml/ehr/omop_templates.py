"""
A series of SQL templates parametrized by workspace, schema, etc. The assumed
SQL dialect is databricks.

Throughout these templates there are a few common variables:
    - `workspace` is the user controlled top level namespace
    - `schema` is the namespace in which to create tables
    - `source` is the data source top level namespace
    - `omop` is the data source's OMOP table namespace

Some tables also include filters on person birthdates `birth_datetime` (this is
an error detection device) and data entry date `start_date`.

A few of the generated SQL scripts may rely on VUMC specific extensions to the
OMOP model or other site-specific quirks; these are identifiable by an "x_"
prepended to a table or field name. Notably the medications table uses
extension fields.

The basic workflow is:

1. Create data tables for conditions, measurements, and medications, with very
   basic filters on data sanity. The measurements table relies on a precomputed
   table of statistics over the measurements.
2. Use the filtered data to count how many times each concept is mentioned and
   put concepts with sufficient data (1000 mentions) into a table.
3. Use the filtered data to count how many unique dates are in each person's
   EHR and put persons with sufficient data (at least 2 unique dates in some
   data channel) into a table.
4. Select all the filtered data, further filtered by the generated concept and
   person tables, as 4-tuples of (person_id, date, concept_id, value).
   Currently only measurements actually use the value field; the other data
   modes include it for compatibility in UNION clauses and in the
   `EHR.dtype` datatype (see ./dtypes.py).
5. Select metadata from the OMOP source for each selected concept.
6. Select demographics (metadata) from the OMOP source for each cohort member.
"""
import argparse
import pathlib
import pprint


CREATE_SCHEMA = """\
CREATE SCHEMA {workspace}.{schema};
"""


CREATE_CONDITIONS = """\
-- Create the Conditions base data for curve construction
DROP TABLE IF EXISTS {workspace}.{schema}.conditions;

CREATE TABLE {workspace}.{schema}.conditions AS (
    SELECT DISTINCT
        P.person_id,
        B.condition_start_date AS date,
        C.concept_id
    FROM {source}.{omop}.person P
    JOIN {source}.{omop}.condition_occurrence B ON (B.person_id = P.person_id)
    JOIN {source}.{omop}.concept C ON (C.concept_id = B.condition_concept_id)
    WHERE (
        P.birth_datetime >= '{birth_datetime}'
        AND B.condition_start_date >= '{start_date}'
        AND C.domain_id = 'Condition'
        AND C.vocabulary_id = 'SNOMED'
    )
)
    ;
"""


CREATE_MEDICATIONS = """\
-- Create the medications base EHR
DROP TABLE IF EXISTS {workspace}.{schema}.medications;

CREATE TABLE {workspace}.{schema}.medications AS (
    SELECT DISTINCT
        P.person_id,
        D.drug_exposure_start_date AS date,
        C.concept_id
    FROM {source}.{omop}.person P
    JOIN {source}.{omop}.drug_exposure D ON (D.person_id = P.person_id)
    JOIN {source}.{omop}.x_drug_exposure X ON (X.drug_exposure_id = D.drug_exposure_id)
    JOIN {source}.{omop}.concept_ancestor CA  ON (CA.descendant_concept_id = D.drug_concept_id)
    JOIN {source}.{omop}.concept C ON (C.concept_id = CA.ancestor_concept_id)
    WHERE (
        P.birth_datetime >= '{birth_datetime}'
        AND D.drug_exposure_start_date >= '{start_date}'
        AND C.concept_class_id = 'Ingredient'
        AND (nullif(X.x_strength, '') IS NOT NULL OR nullif(X.x_dose, '') IS NOT NULL)
        AND nullif(X.x_frequency, '') IS NOT NULL
        AND D.route_source_value IS NOT NULL
        AND X.x_doc_stype = 'Problem list'
    )
)
    ;
"""


CREATE_MEASUREMENT_STATS = """\
-- Creates table of measurement statistics for use in channel selection.
DROP TABLE IF EXISTS {workspace}.{schema}.measurement_stats;

CREATE TABLE {workspace}.{schema}.measurement_stats AS
SELECT
    M.measurement_concept_id,
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
    concept_class_id = 'Lab Test'
    OR concept_class_id = 'Clinical Observation'
    OR concept_id = '44806420'
)
GROUP BY
    M.measurement_concept_id,
    C.concept_class_id,
    C.vocabulary_id,
    C.concept_code,
    C.concept_name
HAVING p50 IS NOT NULL AND p25 != p75
ORDER BY count DESC
    ;
"""



CREATE_MEASUREMENTS = """\
-- Requires table measurement_stats
DROP TABLE IF EXISTS {workspace}.{schema}.measurements;

CREATE TABLE {workspace}.{schema}.measurements AS (
    SELECT DISTINCT
        P.person_id,
        M.measurement_date AS date,
        S.measurement_concept_id AS concept_id,
        last_value(M.value_as_number) OVER (
            PARTITION BY P.person_id, S.measurement_concept_id, M.measurement_date
            ORDER BY M.measurement_datetime ROWS
            BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS value
    FROM {source}.{omop}.person P
    JOIN {source}.{omop}.measurement M ON (M.person_id = P.person_id)
    JOIN {workspace}.{schema}.measurement_stats S ON (S.measurement_concept_id = M.measurement_concept_id)
    WHERE (
        P.birth_datetime >= '{birth_datetime}'
        AND M.measurement_date >= '{start_date}'
        AND M.value_as_number IS NOT NULL
        -- Outlier / error filtration
        AND (S.p01 <= 0 OR M.value_as_number > 0)
        AND (M.value_as_number < S.high_threshold)
        -- Remove specific non-lab LOINCs
        AND S.concept_code != '29463-7'     -- Body Weight
        AND S.concept_code != '8302-2'      -- Body Height
        AND S.concept_code != '39156-5'     -- Body Mass index
    )
)
    ;
"""


CREATE_CORE_CONCEPTS = """\
-- Filter the concepts in the three major OMOP data modes we use (conditions,
-- measurements, medications) to those which have certain numbers of mentions
DROP TABLE IF EXISTS {workspace}.{schema}.core_concepts;

CREATE TABLE {workspace}.{schema}.core_concepts AS (
    SELECT DISTINCT concept_id
    FROM {workspace}.{schema}.conditions
    GROUP BY concept_id
    HAVING count(*) >= 1000
        UNION
    SELECT DISTINCT concept_id
    FROM {workspace}.{schema}.medications
    GROUP BY concept_id
    HAVING count(*) >= 1000
        UNION
    SELECT DISTINCT concept_id
    FROM {workspace}.{schema}.measurements
    GROUP BY concept_id
    HAVING count(*) >= 1000 AND count(DISTINCT person_id) >= 10
)
    ;
"""


CREATE_CORE_COHORT = """\
-- Require each person_id in our cohort to have at least 2 dates in at least
-- one channel from at least one of the three major data modes. I.e., exclude
-- all persons who have *only* constant curves or lack enough dates of data to
-- build curves at all.
DROP TABLE IF EXISTS {workspace}.{schema}.core_cohort;

-- Variations:
--      1. `INTERSECT` instead of `UNION` (pt must have data in _all_ modes)
--      2. group by just person_id (pt can have only constant curves)
--      3. group after union (similar but slightly different effect as 2)
CREATE TABLE {workspace}.{schema}.core_cohort AS (
    SELECT DISTINCT person_id
    FROM {workspace}.{schema}.conditions
    JOIN {workspace}.{schema}.core_concepts USING (concept_id)
    GROUP BY concept_id, person_id
    HAVING count(DISTINCT date) >= 2
        UNION
    SELECT DISTINCT person_id
    FROM {workspace}.{schema}.medications
    JOIN {workspace}.{schema}.core_concepts USING (concept_id)
    GROUP BY concept_id, person_id
    HAVING count(DISTINCT date) >= 2
        UNION
    SELECT DISTINCT person_id
    FROM {workspace}.{schema}.measurements
    JOIN {workspace}.{schema}.core_concepts USING (concept_id)
    GROUP BY concept_id, person_id
    HAVING count(DISTINCT date) >= 2
)
    ;
"""


SELECT_CORE_DATA = """\
WITH core_data AS (
    SELECT DISTINCT person_id, date, concept_id, NULL AS value
    FROM {workspace}.{schema}.conditions
        UNION
    SELECT DISTINCT person_id, date, concept_id, NULL AS value
    FROM {workspace}.{schema}.medications
        UNION
    SELECT DISTINCT person_id, date, concept_id, value
    FROM {workspace}.{schema}.measurements
)
SELECT DISTINCT
    person_id,
    concept_id,
    date::TIMESTAMP_NTZ AS datetime,
    value::DOUBLE
FROM core_data
JOIN {workspace}.{schema}.core_concepts USING (concept_id)
JOIN {workspace}.{schema}.core_cohort USING (person_id)
ORDER BY person_id, concept_id, datetime
    ;
"""


SELECT_CORE_CONCEPT_META = """\
-- Gets concept metadata, including our data mode annotations and (importantly)
-- the fill values per data mode (0 for meds, 1/(20*365.25) for codes, and the
-- 50th percentile for labs). The fill value for conditions is the "background
-- intensity," the value we would expect a random event to have if it showed up
-- once in a 20-year period.

WITH mode_fill_values AS (
    SELECT DISTINCT
        measurement_concept_id AS concept_id,
        p50 AS fill_value,
        'Measurement' AS data_mode
    FROM {workspace}.{schema}.measurement_stats
    UNION
    SELECT DISTINCT
        concept_id,
        0.00013689253935660506 AS fill_value,
        'Condition' AS data_mode
    FROM {workspace}.{schema}.conditions
    UNION
    SELECT DISTINCT
        concept_id,
        0.0 AS fill_value,
        'Medication' AS data_mode
    FROM {workspace}.{schema}.medications
)
SELECT
    C.concept_id,
    C.concept_name,
    C.domain_id,
    C.vocabulary_id,
    C.concept_class_id,
    C.concept_code,
    M.data_mode,
    M.fill_value::DOUBLE
FROM {workspace}.{schema}.core_concepts
JOIN {source}.{omop}.concept C USING (concept_id)
JOIN mode_fill_values M USING (concept_id)
ORDER BY M.data_mode, C.concept_id
    ;
"""


# See the comments below for ref about source concept ids. Particularly, the
# coalesce is to catch any NULL that snuck past the ETL crew and make sure
# they're mapped to the appropriate VUMC Gender / Race vocabulary "unknown"
# concept ids.
SELECT_CORE_COHORT_DEMOGRAPHICS = """\
-- Get, for each person in the core cohort, (a) birthdate, (b) sex, (c) race
-- We include both the source value and concept id; the source values can vary
-- slightly across implementations; its better to process in Py than SQL.

SELECT DISTINCT
    person_id,
    birth_datetime::DATE AS birth_date,
    gender_concept_id,
    GC.concept_name AS gender_concept_name,
    COALESCE(gender_source_concept_id, 2000003109) AS gender_source_concept_id,
    gender_source_value,
    race_concept_id,
    RC.concept_name AS race_concept_name,
    COALESCE(race_source_concept_id, 2003603112) AS race_source_concept_id,
    race_source_value
FROM {workspace}.{schema}.core_cohort
JOIN {source}.{omop}.person P USING (person_id)
JOIN {source}.{omop}.concept GC ON (P.gender_concept_id = GC.concept_id)
JOIN {source}.{omop}.concept RC ON (P.race_concept_id = RC.concept_id)
    ;
"""

# XXX: Non-OMOP: OMOP's take on Race/Gender is unusable.
# (1) - they map "unknown" for both race and gender to the "Null" concept,
# concept_id 0 ("No matching concept"). This is correct sort of, but means that
# the concept_id is no longer independently interpretable; If I use concept ids
# as feature labels I don't have independent data channels for "unknown race"
# and "unknown sex".
# (2) Furthermore, and much worse, they refuse to provide Race concepts
# differentiating Unknown, Refused to Answer, Multiple Answers, and Other. All
# of these (by our current ETL) wind up mapped to the null concept, because
# that is what OMOP wants. This is unusable.
#
# Therefore, we fall back on the original VUMC vocabularies for Gender and
# Race, for which our upstream ETL people have provided local concept_ids in
# the > 2_000_000_000 range (all concepts above 2_000_000_000 are "local" and
# gauranteed not to conflict with standard concepts in the OMOP CDM).
SELECT_DEMOGRAPHIC_META = """\
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
"""


TEMPLATES = dict(
    CREATE_SCHEMA=CREATE_SCHEMA,
    CREATE_CONDITIONS=CREATE_CONDITIONS,
    CREATE_MEDICATIONS=CREATE_MEDICATIONS,
    CREATE_MEASUREMENT_STATS=CREATE_MEASUREMENT_STATS,
    CREATE_MEASUREMENTS=CREATE_MEASUREMENTS,
    CREATE_CORE_CONCEPTS=CREATE_CORE_CONCEPTS,
    CREATE_CORE_COHORT=CREATE_CORE_COHORT,
    SELECT_CORE_DATA=SELECT_CORE_DATA,
    SELECT_CORE_CONCEPT_META=SELECT_CORE_CONCEPT_META,
    SELECT_CORE_COHORT_DEMOGRAPHICS=SELECT_CORE_COHORT_DEMOGRAPHICS,
    SELECT_DEMOGRAPHIC_META=SELECT_DEMOGRAPHIC_META,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=pathlib.Path, default=None)
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--schema', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--omop', type=str)
    parser.add_argument('--debug', action='store_true')
    # A filter on patient birthdates. Used mostly to exclude erroneous entries.
    parser.add_argument('--birth-datetime', default='1920-01-01')
    # A filter on entry date of data
    parser.add_argument('--start-date', default='2000-01-01')
    args = parser.parse_args()
    pprint.pprint(vars(args))

    # Print the *templates*, without any filling, straight to stdout
    if args.debug:
        for t in TEMPLATES.values():
            print(t)
        exit()

    # Exclude None vals so that the templates will throw KeyError if they are
    # missing arguments
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    outdir = kwargs.pop('outdir', None)
    # Write to stdout if there is no outdir
    if outdir is None:
        for t in TEMPLATES.values():
            print(t.format(**kwargs))
        exit()

    assert outdir.exists()
    for name, t in TEMPLATES.items():
        (outdir/f'{name.lower()}.sql').write_text(t.format(**kwargs))
