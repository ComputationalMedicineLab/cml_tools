"""Script to generate and manipulate curves over EHR data"""
import logging
import os
import pickle
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Lock, Process, Queue
from multiprocessing.managers import SharedMemoryManager
from operator import itemgetter
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from time import perf_counter

import numpy as np

from cml import init_logging, iter_batches, get_nproc
from cml.ehr.curves import (build_age_curve,
                            build_demographic_curves,
                            build_ehr_curves,
                            calculate_age_stats,
                            select_cross_section,
                            CurvePointSet)
from cml.ehr.dtypes import Cohort, ConceptMeta, EHR
from cml.ehr.samplespace import SampleIndex, SampleSpace
from cml.ehr.standardizer import make_log10scaler, Log10Scaler
from cml.script.merge_samples import build_output
from cml.stats.incremental import IncrStats


N_PROC = get_nproc()
N_BYTES = 2147483648


def worker(proc_num, input_batch, mem, lock, queue, meta, shapes, datedict, args):
    n = len(input_batch)
    w = len(str(n))
    log_id = f'[{os.getpid()}][{proc_num:3}]'

    messages = []
    samples = []
    misses = []
    total_samples_taken = 0

    # initial calls to some CPU clocks can take a second; this is to "warm
    # up" the CPU timer so the first pairs of timings isn't slow
    perf_counter()

    data = EHR.wrap_shared_memory(shapes, mem)
    for total, (person_id, (i, j), (dmin, dmax)) in enumerate(input_batch, start=1):
        # Select the patient EHR and verify its all for one patient. If
        # this assert fails then check that the source data is sorted.
        group = data[i:j]
        assert np.all(person_id == group.person_id), person_id

        # Evaluate each date in the given datedict, using data only from the
        # record start (dmin) to the date given (dt)
        for dt in datedict[person_id]:
            if not (dt > dmin):
                misses.append((person_id, dt))
                continue
            curveset = build_ehr_curves(group, meta, start=dmin, until=dt)
            point = select_cross_section(curveset, [dt])
            samples.append(pickle.dumps(point.astuple))
            total_samples_taken += 1

        if messages and lock.acquire(block=False):
            try:
                for msg in messages:
                    logging.info(msg)
                messages = []
            finally:
                lock.release()
        elif total > 0 and total % (n//10) == 0:
            msg = f'{log_id} Processed {total/n:3.0%} ({total:{w}} / {n})'
            msg += f'; samples taken: {total_samples_taken:,}'
            messages.append(msg)
    # Do last: parent process knows it can `join` when we `put` on this queue.
    queue.put((proc_num, samples, misses))


def cli():
    """Generate curves from core EHR data and place in an output file"""
    parser = ArgumentParser(description=__doc__,
                            fromfile_prefix_chars='@',
                            formatter_class=RawTextHelpFormatter)
    f = parser.add_argument

    ### Basic input / output file arguments
    f('--data', type=Path, required=True,
      help='Input .npz file of structured base EHR data (cf. cml.ehr.dtypes)')

    f('--meta', type=Path, required=True,
      help='Input .pkl list of ConceptMeta tuples (cf. cml.ehr.dtypes)')

    f('--demo', type=Path, required=True,
      help='Input .pkl list of demographic ConceptMeta tuples')

    # Reference data parameters: needed to scale the matrix before projection
    f('--cohort', type=Path, required=True,
      help='PersonMeta objects for the persons to be projected')

    f('--ref-cohort', type=Path, required=True,
      help='Reference sequence of PersonMeta tuples (for age stats)')

    f('--ref-stats', type=Path, required=True,
      help='Reference stats over the dataset used to preprocess projections')

    f('--ref-space', type=Path, required=True,
      help='SampleSpace over the model discovery data')

    # The set of person_id, date pairs to evaluate for projection
    f('--sampleindex', type=Path, required=True,
      help='Input SampleIndex of (person_id, date) pairs to evaluate.')

    # Output files
    f('--samplesfile', type=Path, required=True,
      help='Output .pkl file for sample points.')

    f('--missesfile', type=Path, required=True,
      help='Output .pkl file for the missing or un-evaluable sample points')

    f('--matrixfile', type=Path,
      help='Output .npy for the merged matrix to project through an ICA model')

    f('--outputindex', type=Path,
      help='Output .npz with the real SampleIndex (excluding misses)')

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
        each subprocess tries to flush the samples to --samplesfile. Subprocesses
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

    # The output files may be *very* large and require a *lot* of time to
    # calculate; so, let us be sure that we want to clobber them if any exist.
    if not args.clobber:
        if args.samplesfile and args.samplesfile.exists():
            logging.info(f'Output file {args.samplesfile} exists, aborting')
            exit()

    logging.info('Loading the EHR meta data from %s', args.meta)
    meta = ConceptMeta.from_pickle_seq(args.meta)

    logging.info('Loading projection EHR data from %s', args.data)
    ehr = EHR.from_npz(args.data)

    logging.info('Loading projection SampleIndex from %s', args.sampleindex)
    index = SampleIndex.from_npz(args.sampleindex)

    logging.info('Processing the SampleIndex into a date dictionary')
    datedict = {}
    for (person_id, dt) in index:
        if person_id in datedict:
            datedict[person_id].append(dt)
        else:
            datedict[person_id] = [dt]

    # Remove persons not in the index from the EHR
    logging.info('Truncating EHR to the persons of the SampleIndex')
    ehr = ehr[np.isin(ehr.person_id, np.unique(index.person_id))]

    # Construct a SampleSpace for the data subset
    space = SampleSpace.from_ehr(ehr)
    logging.info(f'Producing {args.nproc} batch(es) of person_ids')
    batches = tuple(iter_batches(space, n=args.nproc))

    with SharedMemoryManager() as smm:
        logging.info('Moving EHR into shared memory')
        mem, _ = ehr.make_shared_memory(smm)

        logging.info(f'Starting {len(batches)} subprocesses')
        lock = Lock()
        queue = Queue()
        procs = []
        for i, id_batch in enumerate(batches):
            procs.append(Process(target=worker, daemon=True, args=(
                i, id_batch, mem, lock, queue, meta, ehr.arr_shapes, datedict, args
            )))
            procs[-1].start()

        results = []
        misses = []
        for _ in procs:
            i, samples, _misses = queue.get()
            results.extend(samples)
            misses.extend(_misses)
            procs[i].join()
        queue.close()

    logging.info('Finalizing sample sets and writing to output files')
    with open(args.samplesfile, 'wb') as file:
        file.writelines(results)

    misses = sorted(set(misses))
    with open(args.missesfile, 'wb') as file:
        pickle.dump(misses, file, protocol=pickle.HIGHEST_PROTOCOL)

    mask = np.array([t in misses for t in index], dtype=bool)
    index = index[~mask]
    index.to_npz(args.outputindex, compress=True)

    # Time to merge the samples and then scale them using the discovery date
    logging.info('Beginning to merge and expand')
    ref_space = SampleSpace.from_npz(args.ref_space)
    n_obs = ref_space.ntimepoints.astype(int)

    demo_meta = ConceptMeta.from_pickle_seq(args.demo)
    demo_concepts = np.unique([m.concept_id for m in demo_meta])

    modes = ConceptMeta.make_mode_mapping(meta)
    fills = {m.concept_id: m.fill_value for m in meta}
    labels = np.sort(np.array([m.concept_id for m in meta]))

    stats = np.load(args.ref_stats)
    base_stats = IncrStats(stats['base_labels'], *stats['base_stats'])
    lg10_stats = IncrStats(stats['lg10_labels'], *stats['lg10_stats'])
    scaler = make_log10scaler(base_stats, lg10_stats, modes, n_obs)

    cohort = Cohort.from_pickle(args.cohort)
    ref_cohort = Cohort.from_pickle(args.ref_cohort)
    age_mean, age_sdev = calculate_age_stats(ref_space, ref_cohort)

    features = np.concatenate(([3022304], demo_concepts, labels))
    n_demos = 1 + len(demo_concepts)

    t0 = perf_counter()
    samples = sorted((pickle.loads(s) for s in results), key=itemgetter(0))
    dt = perf_counter() - t0
    logging.info('Reconstituted Samples in %.2f', dt)

    shape = (len(features), len(index))
    logging.info(f'Initializing new memory buffer: {shape=}')
    buffer = np.empty(shape)
    t0 = perf_counter()
    X = buffer[n_demos:]
    build_output(index, samples, labels, fills, scaler, out=X)
    logging.info('Calculated data expansion and scaling: (%d, %d) in %.2f',
                 X.shape[0], X.shape[1], perf_counter() - t0)
    del X

    t0 = perf_counter()
    ac = build_age_curve(index, cohort, age_mean, age_sdev)
    if len(ac) != len(index):
        raise RuntimeError('Age curve does not match index')
    dc = build_demographic_curves(index, cohort, demo_concepts)
    if not (dc.shape == (len(demo_concepts), len(buffer[0]))):
        raise RuntimeError(f'{dc.shape=} failed to match')
    logging.info('Calculated demo curves in %.2f', perf_counter()-t0)

    t0 = perf_counter()
    np.save(args.matrixfile, buffer)
    logging.info('Persisted results in %.2f', perf_counter() - t0)
    logging.info('DONE')


if __name__ == '__main__':
    cli()
