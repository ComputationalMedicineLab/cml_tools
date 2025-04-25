# Miscellaneous utilities and small funcs can go here
# TODO: make a cml.formats module for some of this junk?
import collections
import itertools
import logging
import os
import pickle
import sys

import numpy as np
import numpy.lib.format as npf
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


def get_nproc(per=2):
    """"Estimate the number of physical cores available to the process"""
    # function process_cpu_count added in 3.13
    if sys.version_info < (3, 13):
        return len(os.sched_getaffinity(0)) // per
    return os.process_cpu_count() // per


def iter_batches(iterable, n=None):
    """
    Yields `n` batches containing approximately equal numbers of items from
    iterable. `n` defaults to an estimate of the number of physical cores the
    process is allowed to utilize.
    """
    if n is None:
        n = get_nproc()
    if n > 1:
        m, k = divmod(len(iterable), n)
        m += int(k > 0)
        yield from itertools.batched(iterable, m)
    else:
        yield (iterable, )


def pickle_stream(obj_stream, filename, mode='wb'):
    """Serialize all objects in `obj_stream` to `filename` using pickle"""
    with open(filename, mode) as file:
        for obj in obj_stream:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_stream(filename):
    """Deserialize all pickles in a file (inverse of pickle_stream)"""
    with open(filename, 'rb') as file:
        while True:
            try:
                obj = pickle.load(file)
            except EOFError:
                break
            else:
                yield obj


def npy_peek(filename):
    """Returns the shape, fortran order, and dtype of an .npy file"""
    npy_header = collections.namedtuple('npy_header', 'shape f_order dtype')
    with open(filename, 'rb') as file:
        match (magic := npf.read_magic(file)):
            case (1, _): header = npf.read_array_header_1_0(file)
            case (2, _): header = npf.read_array_header_2_0(file)
            case _: raise ValueError(f'cannot parse {filename=} {magic=}')
    return npy_header(*header)


def wrapdtypes(*codes):
    """Make sure all input arguments are well behaved np.dtype objects"""
    return tuple(map(np.dtype, codes))


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
