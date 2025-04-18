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

from cml import init_logging, get_nproc
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
    del n, w

    data = EHR.wrap_sharedmem(mem, shape).data
    batch = []
    nbytes = 0
    stats = None
    # initial calls to some CPU clocks can take a second; this is to "warm
    # up" the CPU timer so the first pairs of timings isn't slow
    perf_counter()

    _write_stats = args.statsfile is not None
    _write_curves = args.curvesfile is not None
    _write_samples = args.samplesfile is not None

    for total, (person_id, i, j) in enumerate(input_batch, start=1):
        # Select the patient EHR and verify its all for one patient. If
        # this assert fails then check that the source data is sorted.
        group = data[i:j]
        assert np.all(person_id == group.person_id)

        if _write_stats:
            curveset, stats = run_curves(group, meta, stats=stats)
        else:
            curveset = build_ehr_curves(group, meta)

        if _write_curves:
            # Spend cycles pickling data *prior* to acquiring the lock.
            batch.append(pickle.dumps(compress_curves(curveset).astuple))
            nbytes += len(batch[-1])
            # If the curve bytestrings exceed the amount of space the process
            # is supposed to use, then try to get the lock and flush the
            # results to file. If we're using more than the hard cap, block.
            block_time = perf_counter()
            if (
                nbytes >= args.maxbytes
                and lock.acquire(block=(nbytes >= 4*args.maxbytes))
            ):
                try:
                    block_time = perf_counter() - block_time
                    logging.info(msg, len(batch), total, nbytes, block_time)
                    with open(args.curvesfile, 'ab') as file:
                        file.writelines(batch)
                    batch = []
                    nbytes = 0
                finally:
                    lock.release()

    # Must remember to write the final batch of curves to disk. Can block now
    # indefinitely, nothing left to do but write and put the stats on the
    # return queue.
    if _write_curves:
        block_time = perf_counter()
        with lock:
            block_time = perf_counter() - block_time
            logging.info(msg, len(batch), total+1, nbytes, block_time)
            with open(args.curvesfile, 'ab') as file:
                file.writelines(batch)

    # Do last: parent process knows it can `join` when we `put` on this queue.
    queue.put((process_num, stats))


def cli():
    """Generate curves from core EHR data and place in an output file"""
    parser = argparse.ArgumentParser(description=__doc__)
    f = parser.add_argument

    ### Basic input / output file arguments
    f('-d', '--datafile', type=Path, default='ehr_data.npy',
      help='Input .npy file of structured base EHR data (see cml.ehr.dtypes)')

    f('-m', '--metafile', type=Path, default='ehr_meta.pkl',
      help='Input .pkl list of ConceptMeta tuples (see cml.ehr.dtypes)')

    f('-o', '--curvesfile', type=Path, default=None,
      help='Output .pkl file for curves (if omitted, curves are not written)')

    f('-s', '--statsfile', type=Path, default=None,
      help='Output .npz file for stats (if omitted, stats are not collected)')

    ### Arguments related to evaluating cross sections from the curveset
    f('-S', '--samplespec', nargs='*', action='append', default=None,
      help="""\
        A string specifying a sampling strategy. May be provided multiple times
        to specify selecting multiple sets of samples. Each argument (or set of
        arguments) must specify 'SOURCE NUMBER DEST'. `SOURCE` may be the
        string token "default" or a path to a pickle to be read by SampleSpace.
        If "default," then the sample space is over the entire underyling EHR.
        If `NUMBER` is the token "None" then the `SOURCE` argument is
        interpreted as providing an explicit list of (person_id, date) points
        to evaluate; otherwise, `NUMBER` samples are selected from the
        corresponding SampleSpace. `DEST` is an output file to which to pickle
        the results. `DEST` is opened in `"ab"` mode and, if more than one
        sample spec is given, in serial, so that the results of many samplings
        could be aggregated as one pickle string.
      """)

    # TODO: implement all these sampling options.
    f('--samplespace', type=Path, default=None, action='append',
      help='Input .pkl with a SampleSpace from which to sample curve points.'
           ' May be specified multiple times to sample in parallel.')

    f('--nsamples', type=int, default=None, action='append',
      help='Number of samples to pull from a SampleSpace if provided')

    f('--samplepoints', type=Path, default=None, action='append',
      help='Input .pkl with a list of patient-datetime points to evaluate;'
           ' only provide one of --samplespace or --samplepoints')

    f('-z', '--samplesfile', type=Path, default=None, action='append',
      help='Output .pkl for sample points evaluated from the curves')

    ### Logging, verbosity, resource control, etc.
    f('-l', '--logfile', type=Path, default=None,
      help='Output file for logging (if omitted and verbose, logs are written'
           ' to stdout; if provided but not verbose, logging is set to INFO)')

    f('-p', '--nproc', type=int, default=N_PROC,
      help='Number of processor cores to use. The sample space is divided'
           ' equally by person_id among all subprocesses. Defaults to an '
           ' estimate of the number of available physical cores')

    f('-b', '--maxbytes', type=int, default=N_BYTES,
      help='Max RAM (in bytes) before each subprocess tries to flush results.'
           ' Each subprocess will block at 4x this amount. Default 2 GiB')

    f('-v', '--verbose', action='count', default=0,
      help='Sets the logging level (-v = INFO, -vv = DEBUG)')

    f('-c', '--clobber', action='store_true', default=False,
      help='If True and argument --datafile exists, overwrite it')

    args = parser.parse_args()
    if args.verbose > 0 or args.logfile:
        level = logging.DEBUG if args.verbose > 1 else logging.INFO
        init_logging(filename=args.logfile, level=level)
    logging.info(pprint(vars(args)))
    #breakpoint()

    # The output file may be *very* large and require a *lot* of time to
    # calculate; so, let us be sure that we want to clobber it if it exists.
    if args.curvesfile.exists():
        if not args.clobber:
            logging.info('Output file exists, aborting')
            exit()
        open(args.curvesfile, 'w').close()

    logging.info('Loading the EHR meta data from %s', args.metafile)
    meta = ConceptMeta.from_pkl(args.metafile)

    logging.info('Loading main EHR from %s', args.datafile)
    ehr = EHR.from_npy(args.datafile).data

    # If we are writing curves or collecting stats this is the thing to do.
    # TODO: we're doing neither, but need samples evaluated.
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
