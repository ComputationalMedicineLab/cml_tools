"""Script to generate and manipulate curves over EHR data"""
import argparse
import logging
import os
import pickle
import warnings
from functools import reduce
from multiprocessing import Lock, Process, Queue
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from pprint import pprint
from time import perf_counter

import numpy as np

from cml import get_nproc
from cml.ehr.curves import build_ehr_curves, compress_curves
from cml.ehr.dtypes import ConceptMeta, EHR
from cml.ehr.samplespace import SampleSpace
from cml.stats.incremental import collect, merge

# If the process has already been restricted to the right number of physical
# cores, pass --nproc=N on the CLI... this is just a fallback.
N_PROC = get_nproc()

# By default, subprocesses try to flush results once 2 Gib RAM accumulated.
# Also by default, each child process will keep working past this limit until
# the file write lock becomes available, up to a (hardcoded) hard limit of 4x
# max bytes, when the child process will block until the lock is available.
# Note that this is not the total amount of memory used by the child process,
# just the total size of the byte (pickled curve objects) it is storing until
# the file becomes available for writing.
N_BYTES = 2147483648

# We've moved to using managing Processes directly (a Pool doesn't seem to gain
# anything since we make exactly one Process per input batch anyway) but I'm
# leaving this comment and function here as documentation about how locks work
# differently between Pools and Processes.
#
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
    results = build_ehr_curves(data, meta, window=window)
    # The base stats should *never* be NaN
    base_stats = collect(results.curves, results.concepts,
                         nansafe=False, nansqueeze=False)
    assert not base_stats.anynan
    # We want to ignore np.log10 divide by zero RuntimeWarning, but not any
    # other warnings that numpy might toss out about invalid math.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log10X = np.log10(results.curves)
    log10X[~np.isfinite(log10X)] = np.nan
    lg10_stats = collect(log10X, results.concepts)
    # If stats already exist, update them.
    if stats is not None:
        base_stats = merge(stats[0], base_stats)
        lg10_stats = merge(stats[1], lg10_stats)
    return results, (base_stats, lg10_stats)


def worker(process_num, input_batch, mem, lock, queue, meta, shape, args):
    n = len(input_batch)
    w = len(str(n))
    msg = (f'[{process_num:3} - {os.getpid()}] writing batch %{w}d '
           f'(%{w}d / {n}); nbytes=%d (%d sec lock wait)')

    data = EHR.wrap_sharedmem(mem, shape).data
    batch = []
    total = 0
    nbytes = 0
    stats = None
    # initial calls to some CPU clocks can take a second; this is to "warm
    # up" the CPU timer so the first pairs of timings isn't slow
    perf_counter()

    for person_id, i, j in input_batch:
        # Select the patient EHR and verify its all for one patient. If
        # this assert fails then check that the source data is sorted.
        group = data[i:j]
        assert np.all(person_id == group.person_id)
        curveset, stats = run_curves(group, meta, stats=stats)
        # Spend cycles pickling data *prior* to acquiring the lock.
        batch.append(pickle.dumps(compress_curves(curveset).astuple))
        nbytes += len(batch[-1])
        total += 1
        # If the curve bytestrings exceed the amount of space the process is
        # supposed to use, then try to get the lock and flush the results to
        # file. If we're using more than the hard cap, block.
        block_time = perf_counter()
        if (
            nbytes >= args.maxbytes
            and lock.acquire(block=(nbytes >= 4*args.maxbytes))
        ):
            try:
                block_time = perf_counter() - block_time
                logging.info(msg, len(batch), total, nbytes, block_time)
                with open(args.outfile, 'ab') as file:
                    file.writelines(batch)
                batch = []
                nbytes = 0
            finally:
                lock.release()
    # Must remember to write the final batch of curves to disk. Can block now
    # indefinitely, nothing left to do but write and put the stats on the
    # return queue.
    block_time = perf_counter()
    with lock:
        block_time = perf_counter() - block_time
        logging.info(msg, len(batch), total, nbytes, block_time)
        with open(args.outfile, 'ab') as file:
            file.writelines(batch)
    # This is *last* - after recv the parent joins the process
    queue.put((process_num, stats))


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
    f('--stats-only', action='store_true', default=False,
      help='Estimate stats over curves but do not write curves to file')
    args = parser.parse_args()

    if args.verbose > 0 or args.logfile:
        level = logging.INFO
        if args.verbose > 1:
            level = logging.DEBUG
        logging.basicConfig(filename=args.logfile,
                            level=level,
                            format='%(asctime)s %(name)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info(pprint(vars(args)))
    # The output file may be *very* large and require a *lot* of time to
    # calculate; so, let us be sure that we want to clobber it if it exists.
    if args.outfile.exists():
        if not args.clobber:
            logging.info('Output file exists, aborting')
            exit()
        open(args.outfile, 'w').close()

    logging.info('Loading the EHR meta data from %s', args.metafile)
    meta = ConceptMeta.from_pkl(args.metafile)

    logging.info('Loading main EHR from %s', args.datafile)
    ehr = EHR.from_npy(args.datafile).data

    logging.info('Batching by person_id')
    batches = SampleSpace.from_ehr(ehr).batch_indices(n=args.nproc)

    with SharedMemoryManager() as smm:
        logging.info('Moving EHR into shared memory')
        mem = smm.SharedMemory(ehr.nbytes)
        EHR.wrap_sharedmem(mem, ehr.shape, source=ehr)

        logging.info(f'Starting {len(batches)} subprocesses')
        lock = Lock()
        queue = Queue()
        procs = []
        for i, inputs in enumerate(batches):
            target_args = (i, inputs, mem, lock, queue, meta, ehr.shape, args)
            p = Process(target=worker, args=target_args, daemon=True)
            procs.append(p)
            p.start()
        results = []
        for _ in procs:
            i, stats = queue.get()
            results.append(stats)
            procs[i].join()
        queue.close()

    logging.info('Finalizing curve stats and writing results')
    base_stats, lg10_stats = tuple(zip(*results))
    base_labels, base_stats = reduce(merge, base_stats).asarrays
    lg10_labels, lg10_stats = reduce(merge, lg10_stats).asarrays
    np.savez(args.statsfile,
             base_labels=base_labels, base_stats=base_stats,
             lg10_labels=lg10_labels, lg10_stats=lg10_stats)
    logging.info('DONE')


if __name__ == '__main__':
    cli()
