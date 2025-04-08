"""Script to generate curves over EHR data"""
import argparse
import logging
import gzip
import os
import pickle
import sys
import warnings
from collections import deque
from multiprocessing import Lock
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.pool import Pool
from pathlib import Path
from time import perf_counter

import numpy as np

from cml.desc_stats import collect_stats, merge_stats
from cml.ehr.curves import build_ehr_curves, compress_curves
from cml.ehr.dtypes import *


# The version of this particular tool
__version__ = '2025.04'

# function process_cpu_count added in 3.13
# In either case, we want the number of physical (not logical) cores.  If the
# process has already been restricted to the right number of physical cores,
# pass --nproc=N on the CLI... this is just a fallback.
if sys.version_info < (3, 13):
    N_PROC = len(os.sched_getaffinity(0)) // 2
else:
    N_PROC = os.process_cpu_count() // 2

# By default, subprocesses try to flush results once 2 Gib RAM accumulated.
# Also by default, each child process will keep working past this limit until
# the file write lock becomes available, up to a (hardcoded) hard limit of 4x
# max bytes, when the child process will block until the lock is available.
# Note that this is not the total amount of memory used by the child process,
# just the total size of the byte (pickled curve objects) it is storing until
# the file becomes available for writing.
N_BYTES = 2147483648
EHR_DTYPE = core_ehr_dtype


# Locks are hard: https://stackoverflow.com/a/69913167
# This is apparently an issue with the Pool rather than passing locks as
# arguments. Locks passed as arguments to Process objects are correct; locks
# passed as arguments to Pool workers are incorrect (bc the pool behind the
# scenes is using a Queue to coordinate its workers, and the lock cannot
# survive pickling through the queue).
def init_proc(_lock):
    """Put a Lock into the process global namespace"""
    global lock
    lock = _lock


def run_curves(data, meta, stats=None, window=365):
    # Build the curves, collect basic and log-transform stats
    results = build_ehr_curves(data, meta, window=window)
    basic_stats = collect_stats(results.curves)
    # We want to ignore np.log10 divide by zero RuntimeWarning, but not any
    # other warnings that numpy might toss out about invalid math.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lnX = np.log10(results.curves)
    lnX[~np.isfinite(lnX)] = np.nan
    log10_stats, log10_mask = collect_stats(lnX, nansafe=True, squeeze_nan=True)
    # Update labels, the log transform may make some of the channels all-nan,
    # in which case the channel and its label are dropped from the results
    basic_labels = np.copy(results.concepts)
    log10_labels = basic_labels[log10_mask]
    # If existing stats exist, update them.
    if stats is not None:
        basic_stats, basic_labels = merge_stats(stats[0], basic_stats,
                                                stats[1], basic_labels)
        log10_stats, log10_labels = merge_stats(stats[2], log10_stats,
                                                stats[3], log10_labels)
    return results, (basic_stats, basic_labels, log10_stats, log10_labels)


class CurveWorker:
    def __init__(self, meta, fname, shape, dtype=EHR_DTYPE, maxbytes=N_BYTES):
        self.meta = meta
        self.fname = fname
        self.shape = shape
        self.dtype = dtype
        self.maxbytes = maxbytes
        self.hard_maxbytes = 4 * maxbytes

    def __call__(self, args):
        global lock
        process_num, buff, indices = args
        # TODO: properly set the field widths of the message by the
        # number of persons (len(indices) to process.
        msg = (f'[{process_num:3} - {os.getpid()}] writing batch %6d '
               f'(%6d / {len(indices)-1}); nbytes=%d (%d sec lock wait)')

        data = wrap_sharedmem(buff, self.shape, self.dtype)

        batch = []
        total = 0
        nbytes = 0
        stats = None
        # initial calls to some CPU clocks can take a second; this is to "warm
        # up" the CPU timer so the first pairs of timings isn't slow
        perf_counter()

        for i, j in indices:
            # Select the patient EHR and verify its all for one patient. If
            # this assert fails then check that the source data is sorted.
            group = data[i:j]
            assert np.all(group.person_id[0] == group.person_id)
            curveset, stats = run_curves(group, self.meta, stats=stats)
            # Strip off our bespoke datatype before serialization to disk.
            # Always store data as close to native types as possible.  Also,
            # spend cycles pickling data *prior* to acquiring the lock.
            batch.append(pickle.dumps(tuple(compress_curves(curveset))))
            nbytes += len(batch[-1])
            total += 1
            # If the curve bytestrings exceed the amount of space the process
            # is supposed to use, then try to get the lock and flush the
            # results to file. If we're using more than the hard cap, block.
            block_time = perf_counter()
            if (
                    nbytes >= self.maxbytes
                    and lock.acquire(block=(nbytes >= self.hard_maxbytes))
            ):
                try:
                    # XXX: does open need to seek the end of the file every time?
                    # Once the file is large does that incur overhead? Is there a
                    # way to keep just one open fd shared by all the processes?
                    block_time = perf_counter() - block_time
                    logging.info(msg, len(batch), total, nbytes, block_time)
                    with open(self.fname, 'ab') as file:
                        file.writelines(batch)
                    batch = []
                    nbytes = 0
                finally:
                    lock.release()
        # Must remember to write the final batch of curves to disk.
        # Can block now indefinitely, nothing left to do but write.
        block_time = perf_counter()
        with lock:
            block_time = perf_counter() - block_time
            logging.info(msg, len(batch), total, nbytes, block_time)
            with open(self.fname, 'ab') as file:
                file.writelines(batch)
        return stats


def wrap_sharedmem(mem, shape, dtype, source=None):
    """Wrap a SharedMemory buffer in an ndarray in a recarray"""
    data = np.ndarray(shape, dtype=dtype, buffer=mem.buf)
    if source is not None:
        data[:] = source
    data = np.rec.array(data, dtype=dtype, copy=False)
    return data


def cli():
    """Generate curves from core EHR data and place in an output file"""
    parser = argparse.ArgumentParser(description=__doc__)
    f = parser.add_argument
    f('-x', '--datafile', type=Path, default='ehr_data.npy')
    f('-m', '--metafile', type=Path, default='ehr_meta.pkl')
    f('-o', '--outfile', type=Path, default='ehr_curves.npy')
    f('-s', '--statsfile', type=Path, default='ehr_curve_stats.npz')
    f('-n', '--nproc', type=int, default=N_PROC)
    f('-b', '--maxbytes', type=int, default=N_BYTES)
    f('-v', '--verbose', action='count', default=0)
    f('-l', '--logfile', type=Path, default=None)
    f('-c', '--clobber', action='store_true', default=False)
    args = parser.parse_args()

    if args.verbose > 0 or args.logfile:
        level = logging.INFO
        if args.verbose > 1:
            level = logging.DEBUG
        logging.basicConfig(filename=args.logfile,
                            level=level,
                            format='%(asctime)s %(name)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    # The output file may be *very* large and require a *lot* of time to
    # calculate; so, let us be sure that we want to clobber it if it exists.
    if args.outfile.exists():
        if not args.clobber:
            logging.info('Output file exists, aborting')
            exit()
        open(args.outfile, 'w').close()

    logging.info('Loading the EHR meta data from %s', args.metafile)
    with open(args.metafile, 'rb') as file:
        meta = pickle.load(file)

    logging.info('Loading main EHR from %s', args.datafile)
    ehr = core_ehr_from_file(args.datafile)

    logging.info('Finding the persons and their indices')
    ids, positions = np.unique(ehr.person_id, return_index=True)
    # Here we make sure each subprocess can't accidentally drop the last id its
    # fed, by feeding it start/stop pairs, and that the last id in the list
    # isn't accidentally dropped (because it won't have an end index without
    # appending the len of the ehr). np.array_split will wrap its input in
    # np.array - it will wrap a generator though, so the call to list is needed
    # to consume the zip generator and instantiate the pairs.
    positions = positions.tolist()
    positions.append(len(ehr))
    positions = list(zip(positions, positions[1:]))
    batches = np.array_split(positions, args.nproc)

    logging.info('Starting process pool with %d processes', args.nproc)
    # The SharedMemoryManager context has to outlive the Pool context, or the
    # shared memory manager will first release all the shared buffers, then the
    # processes will whine while they are shutting down. XXX: Since we make
    # exactly as many jobs as there are nprocs anyways (see the construction of
    # the batch indices `batches`), and the pool introduces this complication
    # and the complexity with locks (see `init_proc`), maybe it'd be easier to
    # understand if we dropped the Pool and managed the Processs directly.
    with SharedMemoryManager() as smm:
        with Pool(args.nproc, initializer=init_proc, initargs=(Lock(),)) as pool:
            worker = CurveWorker(meta, args.outfile, ehr.shape,
                                 dtype=ehr.dtype,
                                 maxbytes=args.maxbytes)

            logging.info('Moving EHR into shared memory')
            mem = smm.SharedMemory(ehr.nbytes)
            wrap_sharedmem(mem, ehr.shape, ehr.dtype, source=ehr)

            logging.info('Running subprocesses')
            arguments = [(i, mem, batch) for i, batch in enumerate(batches)]
            results = list(pool.imap_unordered(worker, arguments))

    logging.info('Finalizing curve stats and writing results')
    st, cx, logst, logcx = results[0]
    for (new_st, new_cx, new_logst, new_logcx) in results[1:]:
        st, cx = merge_stats(st, new_st, cx, new_cx)
        logst, logcx = merge_stats(logst, new_logst, logcx, new_logcx)
    np.savez(args.statsfile, basic_stats=st, basic_concepts=cx,
             log10_stats=logst, log10_concepts=logcx)
    logging.info('DONE')


if __name__ == '__main__':
    cli()
