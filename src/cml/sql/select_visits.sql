-- The CTE filters the data points by concept & person tables, and then we get
-- the min/max datetimes of the data so we can clamp the visit start datetimes
-- to the earliest datapoint. This way we avoid "dead space" - if we sample
-- from a visit prior to any data being observed, the sample is empty.
WITH patient_ranges AS (
    WITH datapoints AS (
{cte}
    )
    SELECT
        person_id,
        min(datetime) AS min_datetime,
        max(datetime) AS max_datetime
    FROM datapoints
    GROUP BY person_id
    HAVING min_datetime < max_datetime
)
SELECT
    person_id,
    -- Clamp the beginning of the visit to the earliest datapoint. Combined
    -- with the WHERE clause below this ensure every sample point from the
    -- selected visits has viable data.
    greatest(start_datetime, min_datetime)::TIMESTAMP_NTZ AS start_datetime,
    end_datetime::TIMESTAMP_NTZ AS end_datetime
FROM {workspace}.{schema}.visits
JOIN patient_ranges USING (person_id)
-- Strictly less than: all selected visits will have nonzero duration
WHERE (start_datetime < end_datetime AND min_datetime < end_datetime)
ORDER BY person_id, start_datetime, end_datetime
    ;
