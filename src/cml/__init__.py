# Overall package version
__version__ = '0.0.1'
# Miscellaneous utilities and small funcs can also go here
import logging

import numpy as np
import pandas as pd


LOG_FORMAT = '%(asctime)s %(name)s %(message)s'
LOG_DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'


def init_logging(filename='', **opts):
    """Init logging with preferred formatting options"""
    opts.setdefault('filename', filename)
    opts.setdefault('level', logging.INFO)
    opts.setdefault('format', LOG_FORMAT)
    opts.setdefault('datefmt', LOG_DATE_FORMAT)
    logging.basicConfig(**opts)


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
