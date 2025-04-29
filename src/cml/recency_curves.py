# TODO this API is provisional; need to work on it a bit if the approach works
import numpy as np


def most_recent(target, data, assume_sorted=True,
                label_field='concept_id', target_field='datetime'):
    # First prune the dataset of dates overshooting the target date.
    data = data[data[target_field] <= target]
    # Sorting may be pricy; but the data should be sorted by the time it gets
    # here. The below works assuming sortedness: assume data is lexically
    # sorted by label and then data. Then if locs[i] and locs[i+1] give the
    # first indices respectively for the ith and ith+1 concepts, (locs[i+1]-1)
    # must be index of the record with the max date for the ith concept.
    if not assume_sorted:
        # FIXME: This sort no longer works since we use SOA instead of recarray
        sort = np.argsort(data[[label_field, target_field]])
        data = data[sort]
    _, locs = np.unique(data[label_field], return_index=True)
    locs = np.concatenate((locs[1:], [len(data)])) - 1
    return data[locs]


def recency(target, dates, tau):
    """
    The recency function is defined for a target time `t_target` greater than
    or equal to observation time `t_obs` as:

    >>> r = 1.0 / (1.0 + ((t_target - t_obs) / tau))

    In practice `t_obs` is the most recent time of observation. Mathematically,
    if no such time exists, it may be thought of as `-inf`; then in the limit
    `r` is `0`. I.e., `r` curves fill missing data with `0`. However, this
    function assumes all observations in dates are real times of comparable
    dtype with target, no nan or inf handling is performed.
    """
    return 1.0 / (1.0 + ((target - dates).astype(float) / tau))


def eval_recency_at_point(target, data, tau_map):
    # Target is a (person_id, datetime) tuple
    # Data is the subset of the EHR for the person_id
    # Modes is a mapping from source concept ids to "mode" strings
    assert np.all(data.person_id == target[0])
    recents = most_recent(target[1], data)

    # The sample time is prior to any data (I.e. the visit must extend to
    # before any data... an unusual scenario. Maybe this needs to prohibited?
    # But if we define the allowable samples as any time in an inpatient
    # admission, maybe we need to allow samples prior to there being any actual
    # data).
#    if not recents:
#        rcp = (target[0], target[1], [], [])
#        lcp = (target[0], target[1], [], [])
#        return (rcp, lcp)

    rec_concepts = np.copy(recents.concept_id)
    rec_values = recency(target[1], recents.datetime,
                         np.array([tau_map[c] for c in rec_concepts]))

    lab_mask = np.isfinite(recents.value)
    lab_concepts = np.copy(recents.concept_id[lab_mask])
    lab_values = np.copy(recents.value[lab_mask])

    rcp = (target[0], target[1], rec_concepts, rec_values)
    lcp = (target[0], target[1], lab_concepts, lab_values)
    return (rcp, lcp)


def eval_recency_curves(index, space, ehr, modes):
    """Evaluate "recency" curves.

    The "tau" parameter is 30 for Conditions, 7 for Medications, Procedures,
    and Labs, and Labs are also carried forward.
    """
    tau_map = {k: 30 if v == 'Condition' else 7 for k, v in modes.items()}
    points = []
    for i, target in enumerate(index):
        _, (p_i, p_j), _ = space[space.index_map[target[0]]]
        points.append(eval_recency_at_point(target, ehr[p_i:p_j], tau_map))
        # Really hacking it in here. Hack hack hack
        if i % 10_000 == 0:
            print(i)
    return points
