"""Expand, fill, standardize a collection of CurvePointSet tuples"""
import logging
import pickle
import warnings
from argparse import ArgumentParser, RawTextHelpFormatter
from operator import itemgetter
from pathlib import Path
from pprint import pformat
from textwrap import dedent
from time import perf_counter

import bottleneck as bn
import numpy as np

from cml import init_logging, get_nproc, unpickle_stream
from cml.ehr.curves import (build_age_curve,
                            build_demographic_curves,
                            calculate_age_stats,
                            CurvePointSet)
from cml.ehr.dtypes import Cohort, ConceptMeta, PersonMeta
from cml.ehr.samplespace import SampleIndex, SampleSpace
from cml.ehr.standardizer import make_log10scaler, Log10Scaler
from cml.label_expand import expand
from cml.stats.incremental import IncrStats


def build_output(index, samples, labels, fills, scaler=None, out=None):
    """Expands and fills the samples, maybe scales them as well"""
    check_index, check_labels, X = expand(samples, labels, fills, out=out)

    if not np.all(check_index[0].astype(np.int64) == index.person_id):
        raise RuntimeError('Expansion index person_ids fail to match')

    if not np.all(check_index[1].astype('M8[D]') == index.datetime):
        raise RuntimeError('Expansion index dates fail to match')

    if not np.all(check_labels == labels):
        raise RuntimeError('Expansion channel labels fail to match')

    if scaler is not None:
        # Ignore numpy complaints about log10 if we're scaling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler(X.T, labels)

    # By this point there should never be a NaN in X: it should be filled
    if bn.anynan(X):
        raise RuntimeError('nan values in X!')
    return X


def sampleset(string):
    """Parse strings from command line into index/sample/output paths"""
    index, samples, output = (Path(s).resolve() for s in string.split())
    # Throw FileNotFound if they don't exist
    index.open('rb').close()
    samples.open('rb').close()
    # Throw some kind of error if the output can't be made a writeable dir
    output.mkdir(exist_ok=True, parents=True)
    return index, samples, output


def check_cache(cache, key, missing_func, log):
    """
    Look for a key in cache, provide with missing_func if missing and log
    how long the acquisition took.
    """
    if (result := cache.get(key)) is None:
        t_start = perf_counter()
        cache[key] = result = missing_func()
        dt = perf_counter() - t_start
        log(f"{key=} missing from cache; acquired in {dt:.2f}")
    return result


def cli():
    """Join and transform sets of curve point samples into matrices"""
    parser = ArgumentParser(description=__doc__,
                            fromfile_prefix_chars='@',
                            formatter_class=RawTextHelpFormatter)
    f = parser.add_argument

    f('--cohort', type=Path, required=True,
      help='Input .pkl: a sequence of PersonMeta tuples (see cml.ehr.dtypes)')

    f('--meta', type=Path, required=True,
      help='Input .pkl: a sequence of ConceptMeta tuples (see cml.ehr.dtypes)')

    f('--demo', type=Path, required=True,
      help=dedent("""\
        An input .pkl containing a sequence of ConceptMeta tuples related to
        sex and race. Currently, sex and race are always one hot encoded with a
        fill value of zero, regardless of what the "fill_value" fields of these
        ConceptMeta records may be.
      """))

    f('--stats', type=Path,
      help=dedent("""\
        Input .npz of statistics for use in applying the Log10Scaler
        transformation to the merged data matrix. If omitted, the scaling
        transformation is omitted. The .npz is expected to have keys
        "base_stats", "base_labels", "lg10_stats", and "lg10_labels". Consult
        the output of `cml.script.ehr_curves`.
      """))

    f('--space', type=Path,
      help=dedent("""\
        A SampleSpace from which to get the total number of observable time
        points in the curve space. (May be required in order to construct the
        Log10Scaler when --stats is provided).
      """))

    f('-S', '--samples', action='append', type=sampleset,
      help=dedent("""\
        A string which specifies three paths: "INDEX SAMPLES DEST". INDEX is a
        path to a SampleIndex compatible .npz; SAMPLES is a path to a pickled
        sequence of CurvePointSet compatible tuples; DEST is a writable output
        directory where the constructed X, index, and features will be written
        as "X.npy", "index.npy", and "features.npy".
      """))

    # TODO: fast-track whitening. Don't waste any time goofing around with
    # storing the merged / standardized data to disk, just whiten it
    # immediately and store the Z, K, and X_mean matrices produced by
    # whitening.
    f('--pca-whiten', type=int,
      help=dedent("""\
            *CURRENTLY UNIMPLEMENTED*
        If True, then the merged curves are not themselves persisted; instead,
        they are whitened and projected into a subspace. The whitened matrix Z,
        the whitening matrix K, and the original data matrix feature mean
        values X_mean are all persisted to disk instead of X.
      """))

    ### Logging, verbosity, resource control, etc.
    f('--cache', type=Path, default='_merge_cache.pkl',
      help=dedent("""\
        A location to cache small but expensive intermediate values such as the
        mean and stdev of the cohort Age, etc. The cache is a simple Python
        dict of builtin and np types which can easily be inspected, augmented,
        or removed. If this behavior is not desire use option --no-cache.
      """))
    f('--no-cache', action='store_true', help='Turn off the file cache')

    f('--logfile', type=Path,
      help=dedent("""\
        Output file for logging. If omitted and --verbose, logs are written to
        stdout; if provided but not verbose, log level is set to logging.INFO
      """))

    f('-v', '--verbose', action='count', default=0,
      help='Sets the logging level (-v = INFO, -vv = DEBUG)')

    f('--clobber', action='store_true',
      help='If True and any output file exists, overwrite it')

    args = parser.parse_args()
    if args.verbose > 0 or args.logfile:
        level = logging.DEBUG if args.verbose > 1 else logging.INFO
        init_logging(filename=args.logfile, level=level)
    logging.info(pformat(vars(args)))

    # Too fast to bother caching
    logging.info('Loading channel metadata')
    meta = ConceptMeta.from_pickle_seq(args.meta)
    fills = {m.concept_id: m.fill_value for m in meta}
    labels = np.sort(np.array([m.concept_id for m in meta]))

    logging.info('Loading cohort metadata')
    cohort = Cohort.from_pickle(args.cohort)

    if args.no_cache or not args.cache.exists():
        cache = {}
    else:
        logging.info('Loading cache')
        with open(args.cache, 'rb') as file:
            cache = pickle.load(file)

    if args.no_cache:
        def cache_log(*args, **kwargs):
            pass
    else:
        cache_log = logging.info

    # If the cache changes, then we want to re-persist it; we detect change
    # just by checking if it has the same keys as when loaded / created.
    cache_orig = tuple(cache)

    if args.stats is None:
        scaler = None
    else:
        # If we are scaling, then assemble the Log10Scaler
        def f():
            return SampleSpace.from_npz(args.space).ntimepoints.astype(int)
        n_obs = check_cache(cache, 'n_obs', f, cache_log)

        def f():
            return ConceptMeta.make_mode_mapping(meta)
        modes = check_cache(cache, 'modes', f, cache_log)

        def f():
            stats = np.load(args.stats)
            base_stats = IncrStats(stats['base_labels'], *stats['base_stats'])
            lg10_stats = IncrStats(stats['lg10_labels'], *stats['lg10_stats'])
            scaler = make_log10scaler(base_stats, lg10_stats, modes, n_obs)
            return scaler.astuple
        scaler = Log10Scaler(*check_cache(cache, 'scaler', f, cache_log))

    def f():
        demo_meta = ConceptMeta.from_pickle_seq(args.demo)
        return np.unique([m.concept_id for m in demo_meta])
    demo_concepts = check_cache(cache, 'demo_concepts', f, cache_log)

    # These two in particular are pricy
    def f():
        space = SampleSpace.from_npz(args.space)
        return calculate_age_stats(space, cohort)
    age_mean, age_sdev = check_cache(cache, 'age_stats', f, cache_log)

    if not args.no_cache and tuple(cache) != cache_orig:
        logging.info('Persisting cache')
        with open(args.cache, 'wb') as file:
            pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO: document the Age Concept ID 3022304, its origin, motivation, etc.
    # NOTE: This particular implementation does not make any allowance for
    # heterogeneous features: all output X matrices are assumed to have a
    # common set of feature labels, which is the total set of all possible
    # features from the project source EHR. Any features without data in this
    # sampling are dead and assumed pruned by the PCA-whitening projection
    # immediately prior to ICA. This way we keep a consistent shape within a
    # given "universe" of channels; this simplifies many things.
    t0 = perf_counter()
    features = np.concatenate(([3022304], demo_concepts, labels))
    n_demos = 1 + len(demo_concepts)
    logging.info('Constructed unified label index in %.2f', perf_counter()-t0)

    item0 = itemgetter(0)
    buffer = None
    for index_path, sample_path, output in args.samples:
        t_start = t0 = perf_counter()
        samples = CurvePointSet.from_pickle_stream(sample_path)
        samples = sorted((s.astuple for s in samples), key=item0)
        dt = perf_counter() - t0
        logging.info('Loaded Samples %s in %.2f', sample_path.name, dt)

        t0 = perf_counter()
        index = SampleIndex.from_npz(index_path)
        dt = perf_counter() - t0
        logging.info('Loaded Index %s in %.2f', index_path.name, dt)

        shape = (len(features), len(index))
        if buffer is None or buffer.shape != shape:
            logging.info(f'Initializing new memory buffer: {shape=}')
            buffer = np.empty(shape)

        # X should be a view backed by buffer: memory should not change
        t0 = perf_counter()
        X = buffer[n_demos:]
        build_output(index, samples, labels, fills, scaler, out=X)
        ## Paranoia check:
        #if not np.all(X == buffer[n_demos:]):
        #    raise RuntimeError('Data expansion or scaling has failed')
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
        buffer[0] = ac
        buffer[1:n_demos] = dc
        logging.info('Copied demo curves to buffer %.2f', perf_counter() - t0)

        t0 = perf_counter()
        np.save(output/'X.npy', buffer)
        # This is redundant? It should just be a copy of whats at index_path
        #index.to_npz('./index.npz', compress=True)
        np.save(output/'features.npy', features)
        logging.info('Persisted results in %.2f', perf_counter() - t0)
        logging.info('COMPLETED expansion for %s in %.4f', sample_path.name,
                     perf_counter() - t_start)

    logging.info('DONE')


if __name__ == '__main__':
    # Some CPU perf clocks need "priming"
    perf_counter()
    cli()
