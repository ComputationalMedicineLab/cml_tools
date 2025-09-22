"""Script to generate and manipulate curves over EHR data"""
import logging
import os
import pickle
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from functools import reduce
from multiprocessing import Lock, Process, Queue
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from time import perf_counter

import numpy as np

from cml import init_logging, iter_batches, get_nproc
from cml.ehr.curves import build_ehr_curves, compress_curves, select_cross_section
from cml.ehr.dtypes import Cohort, ConceptMeta, EHR
from cml.ehr.samplespace import SampleIndex, SampleSpace, sample_uniform
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


# TODO:
# 1. Input load balancing: use a Queue to feed person ids to the workers. Not a
#    Pool! The pool will try to manage the workers in ways that are a little
#    obscure. We want them to maintain internal state and make their own
#    decisions about when to, e.g., write results to disk.
# 2. Logging Queue - let the main process do all the logging
# 3. Logging format - use a uniform format *per mode* (curves, stats, samples).
# *** In fact, I might want a _different worker function_ per mode. Tailor made
# worker functions could be faster and easier to understand. What are the valid
# combinations of mode?
# - curves, stats, samples
# - curves, stats
# - curves, samples
# - stats, samples
# - curves
# - stats
# - samples


def samplespec(spec):
    """Custom 'type' for sample spec arguments (cf. ArgumentParser in `cli`).

    Examples:
        $ <invoke_script> -S 'output.pkl sample_space.pkl 600_000'
        $ <invoke_script> -S 'output.pkl sample_index.pkl'
        $ <invoke_script> -S 'output.pkl 600_000'
    """
    match spec.split():
        case out, inp, num:
            num = int(num)
        case out, inp_or_num:
            try:
                num = int(inp_or_num)
            except ValueError:
                inp = inp_or_num
                num = None
            else:
                inp = None
        case _:
            raise ValueError(f'Cannot parse {spec=}')
    out = Path(out).resolve()
    inp = Path(inp).resolve()
    # Throw FileNotFound if the input file doesn't exist
    inp.open('rb').close()
    return out, inp, num


def run_curves(data, meta, stats=None, start=None, until=None, window=365):
    results = build_ehr_curves(data, meta, start=start, until=until, window=window)
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


def worker(proc_num, input_batch, mem, lock, queue, meta, shapes, specs, args):
    n = len(input_batch)
    w = len(str(n))
    log_id = f'[{os.getpid()}][{proc_num:3}]'
    curve_msg = (f'{log_id} writing batch %{w}d '
                 f'(%{w}d / {n}); nbytes=%d (%d sec lock wait)')

    # Percent of input batch to log if we're *not* writing curves
    log_interval = n // 10
    messages = []

    batch = []
    nbytes = 0
    stats = None

    _write_stats = args.statsfile is not None
    _write_curves = args.curvesfile is not None
    samples = [[] for _ in specs]
    total_samples_taken = 0

    # initial calls to some CPU clocks can take a second; this is to "warm
    # up" the CPU timer so the first pairs of timings isn't slow
    perf_counter()

    data = EHR.wrap_shared_memory(shapes, mem)
    for total, (person_id, (i, j), (dmin, dmax)) in enumerate(input_batch, start=1):
        # Select the patient EHR and verify its all for one patient. If
        # this assert fails then check that the source data is sorted.
        group = data[i:j]
        assert np.all(person_id == group.person_id)

        # Check that we are writing curves, collecting stats, or this person
        # has curve points we want to evaluate and collect. If all three are
        # False we can skip to the next person_id in the batch
        if not (_write_stats or _write_curves):
            if not any(person_id in sample_set for sample_set, _ in specs):
                continue

        if _write_stats:
            curveset, stats = run_curves(group, meta, stats=stats)
        else:
            curveset = build_ehr_curves(group, meta, start=dmin, until=dmax)

        for i, (sample_set, _) in enumerate(specs):
            if (dates := sample_set.get(person_id)) is not None:
                points = select_cross_section(curveset, dates)
                samples[i].append(pickle.dumps(points.astuple))
                total_samples_taken += len(dates)

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
                    logging.info(curve_msg, len(batch), total, nbytes, block_time)
                    with open(args.curvesfile, 'ab') as file:
                        file.writelines(batch)
                    batch = []
                    nbytes = 0
                finally:
                    lock.release()
        # If there are any other logs to write *and* we can get the lock, write
        # the logs: but we never block here just for logging. Logs can be late.
        elif messages and lock.acquire(block=False):
            try:
                for msg in messages:
                    logging.info(msg)
                messages = []
            finally:
                lock.release()
        elif total > 0 and total % log_interval == 0:
            msg = f'{log_id} Processed {total/n:3.0%} ({total:{w}} / {n})'
            if specs:
                msg += f'; samples taken: {total_samples_taken:,}'
            messages.append(msg)

    # Must remember to write the final batch of curves to disk. Can block now
    # indefinitely, nothing left to do but write and put the stats on the
    # return queue.
    if _write_curves:
        block_time = perf_counter()
        with lock:
            block_time = perf_counter() - block_time
            logging.info(curve_msg, len(batch), total+1, nbytes, block_time)
            with open(args.curvesfile, 'ab') as file:
                file.writelines(batch)

    # Do last: parent process knows it can `join` when we `put` on this queue.
    queue.put((proc_num, stats, samples))


def cli():
    """Generate curves from core EHR data and place in an output file"""
    parser = ArgumentParser(description=__doc__,
                            fromfile_prefix_chars='@',
                            formatter_class=RawTextHelpFormatter)
    f = parser.add_argument

    ### Basic input / output file arguments
    f('-d', '--datafile', type=Path, required=True,
      help='Input .npz file of structured base EHR data (see cml.ehr.dtypes)')

    f('-m', '--metafile', type=Path, required=True,
      help='Input .pkl list of ConceptMeta tuples (see cml.ehr.dtypes)')

    f('-o', '--curvesfile', type=Path,
      help='Output .pkl file for curves. If omitted, curves are not written.')

    f('-s', '--statsfile', type=Path,
      help='Output .npz file for stats. If omitted, stats are not collected.')

    f('--samplespace', type=Path,
      help='Samplespace to override the start/stop dates of the default space')

    f('-S', '--samplespec', action='append', type=samplespec,
      help=dedent("""\
        A string specifying a sampling strategy. May be provided multiple times
        to specify selecting multiple sets of samples which will all be
        collected in parallel. I.e., each patient's CurveSet will only be
        evaluated once, instead of once per sample set.

        The sample spec must be a string of one of three formats:
            (1) "DEST SRC NUM"
            (2) "DEST SRC"
            (3) "DEST NUM"
        DEST must be a valid, writable filename, where the evaluated sample
        points will be placed. If both SRC and NUM are given, then a
        SampleSpace is loaded from SRC and NUM samples are drawn uniformly at
        random from the sample space. If only SRC is given, a SampleIndex
        is loaded from SRC which specifies exactly the patient dates to sample.
        If only NUM is specified, then NUM samples are draw from the "default"
        sample space (i.e. the entire set of patient-dates defined by the
        source EHR structure: see --datafile).

        Be aware that samples are held in memory until all curves have been
        processed and the subprocesses begin to shut down. Curve points are
        usually so sparse this shouldn't typically be an issue.
      """))

    ### Logging, verbosity, resource control, etc.
    f('-l', '--logfile', type=Path,
      help=dedent("""\
        Output file for logging. If omitted and --verbose, logs are written to
        stdout; if provided but not verbose, log level is set to logging.INFO
      """))

    f('-p', '--nproc', type=int, default=N_PROC,
      help=dedent("""\
        Number of processor cores to use. The sample space is divided equally
        by person_id among all subprocesses. Defaults to an estimate of the
        number of available physical cores.
      """))

    f('-b', '--maxbytes', type=int, default=N_BYTES,
      help=dedent("""\
        Max RAM in bytes (default 2Gb) used to store evaluated curves before
        each subprocess tries to flush the curves to --curvesfile. Subprocesses
        will continue to work up to 4x this amount before blocking; i.e., this
        sets the lower threshold when it will begin attempting to write, and 4x
        this amount is the hard cap. This count **is only** for curves which
        are waiting to be written to disk, it **does not include** memory used
        for sample indices, statistics, evaluated sample points, or any other
        process overhead.
      """))

    f('-v', '--verbose', action='count', default=0,
      help='Sets the logging level (-v = INFO, -vv = DEBUG)')

    f('-c', '--clobber', action='store_true',
      help='If True and any output file exists, overwrite it')

    args = parser.parse_args()
    if args.verbose > 0 or args.logfile:
        level = logging.DEBUG if args.verbose > 1 else logging.INFO
        init_logging(filename=args.logfile, level=level)
    logging.info(pformat(vars(args)))

    if not any((args.curvesfile, args.statsfile, args.samplespec)):
        parser.error('No output for curves, stats, or samples. Nothing to do!')

    # The output files may be *very* large and require a *lot* of time to
    # calculate; so, let us be sure that we want to clobber them if any exist.
    if not args.clobber:
        if args.curvesfile and args.curvesfile.exists():
            logging.info(f'Output file {args.curvesfile} exists, aborting')
            exit()
        for out, _, _ in args.samplespec:
            if out.exists():
                logging.info(f'Output samples file {out} exists, aborting')

    logging.info('Loading the EHR meta data from %s', args.metafile)
    meta = ConceptMeta.from_pickle_seq(args.metafile)

    logging.info('Loading main EHR data from %s', args.datafile)
    ehr = EHR.from_npz(args.datafile)

    if args.samplespace:
        logging.info(f'Loading samplespace from {args.samplespace}')
        default_space = SampleSpace.from_npz(args.samplespace)
    else:
        logging.info('Creating default SampleSpace from the EHR')
        default_space = SampleSpace.from_ehr(ehr)
    logging.info(f'Producing {args.nproc} batch(es) of person_ids')
    batches = tuple(iter_batches(default_space, n=args.nproc))

    # Process specs into a list of 2-tuples (index, outfile). The index in each
    # tuple is a mapping from person_id ints to date arrays. If there are no
    # sampling specs the list is just empty.
    if args.samplespec is not None:
        logging.info(f'Processing {len(args.samplespec)} sample specs')
        specs = []
        spaces = {None: default_space}
        for out, inp, num in args.samplespec:
            logging.debug(f'Spec: {inp=} {num=} {out=}')
            if num is None:
                logging.info(f'Loading SampleIndex from {inp=}')
                sample_index = SampleIndex.from_npz(inp)
            else:
                # Enable sampling the same space repeatedly without reloading
                # it from disk (if inp is None then spaces contains the default
                # space; aka the whole sampling space over the EHR).
                if inp not in spaces:
                    logging.info(f'Loading SampleSpace from {inp=}')
                    space = SampleSpace.from_npz(inp)
                    spaces[inp] = space
                space = spaces[inp]
                logging.info(f'Selecting {num=} points from the space')
                with space.dt_unit_ctx('D'):
                    sample_index = sample_uniform(space, num)
            specs.append((sample_index.mapping, out))
        del spaces, sample_index
    else:
        specs = []

    with SharedMemoryManager() as smm:
        # cf. record.SOA: make_shared_memory and wrap_shared_memory
        logging.info('Moving EHR into shared memory')
        mem, _ = ehr.make_shared_memory(smm)

        logging.info(f'Starting {len(batches)} subprocesses')
        lock = Lock()
        queue = Queue()
        procs = []
        for i, id_batch in enumerate(batches):
            procs.append(Process(target=worker, daemon=True, args=(
                i, id_batch, mem, lock, queue, meta, ehr.arr_shapes, specs, args
            )))
            procs[-1].start()

        result_stats = []
        result_samples = [[] for _ in specs]
        for _ in procs:
            i, stats, samples = queue.get()
            result_stats.append(stats)
            for _list, points in zip(result_samples, samples):
                _list.extend(points)
            procs[i].join()
        queue.close()

    if args.statsfile is not None:
        logging.info('Finalizing curve stats and writing results')
        base_stats, lg10_stats = tuple(zip(*result_stats))
        base_labels, base_stats = reduce(merge, base_stats).asarrays
        lg10_labels, lg10_stats = reduce(merge, lg10_stats).asarrays
        np.savez(args.statsfile,
                 base_labels=base_labels, base_stats=base_stats,
                 lg10_labels=lg10_labels, lg10_stats=lg10_stats)

    # Remember the sampled curve points are stripped of the CurvePointSet class
    # and pickled *immediately* when they are created. This is better for
    # space, has lower Queue transmit overhead, and simplifies writing to disk.
    # So each output file below is a pickle _stream_ of CurvePointSet _tuples_.
    if args.samplespec is not None:
        logging.info('Finalizing sample sets and writing to output files')
        for samples, (_, outfile) in zip(result_samples, specs):
            with open(outfile, 'wb') as file:
                file.writelines(samples)

    logging.info('DONE')


if __name__ == '__main__':
    cli()
