# Miscellaneous utilities and small funcs can go here
import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd


# Overall package version
__version__ = '0.0.1'


LOG_FORMAT = '%(asctime)s %(name)s %(message)s'
LOG_DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'


def init_logging(filename='', **opts):
    """Init logging with preferred formatting options"""
    opts.setdefault('filename', filename)
    opts.setdefault('level', logging.INFO)
    opts.setdefault('format', LOG_FORMAT)
    opts.setdefault('datefmt', LOG_DATE_FORMAT)
    logging.basicConfig(**opts)


def get_nproc(n=2):
    """"Estimate the number of physical cores available to the process"""
    # function process_cpu_count added in 3.13
    if sys.version_info < (3, 13):
        return len(os.sched_getaffinity(0)) // n
    return os.process_cpu_count() // n


def iter_batches(iterable, n=None, consume=True):
    """
    Yields `n` batches containing approximately equal numbers of items from
    iterable. `n` defaults to an estimate of the number of physical cores the
    process is allowed to utilize.
    """
    if n is None: n = get_nproc()
    if consume: iterable = tuple(iterable)
    yield from itertools.batched(iterable, (len(iterable)//n)+1)



# TODO: make a cml.formats module for this kind of junk?
def df_to_np(curves: pd.DataFrame):
    """Decompose a pd.DataFrame to a dict of np.ndarrays"""
    return dict(values=curves.to_numpy(),
                index=curves.to_numpy(),
                columns=curves.to_numpy())


def np_to_df(data_dict: dict[str, np.ndarray], multi_index=True):
    """Reconstruct a pd.DataFrame from a dict of np.ndarrays"""
    index = data_dict['index']
    columns = data_dict['columns']
    if multi_index:
        index = pd.MultiIndex.from_tuples(index)
        columns = pd.MultiIndex.from_tuples(columns)
    return pd.DataFrame(data_dict['values'], index=index, columns=columns)
