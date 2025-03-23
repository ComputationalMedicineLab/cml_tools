import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from cml_tools.fastica import (apply_model, fastica, recover_S_from_WX1,
                               recover_A_from_WK, scale_to_unit_variance)
from cml_tools.whiten import apply_whitening, learn_whitening

# The version of this fastica command line tool
__version__ = '2025.03'
torch.tanh(torch.tensor(0))

def eps_(t):
    return torch.finfo(t.dtype).eps

def pprint_param_ns(ns, prefix=''):
    nsdict = vars(ns)
    fmt = f'>{max(len(k) for k in nsdict)}s'
    for key, val in sorted(nsdict.items()):
        logging.info('%s %s: %s', prefix, format(key, fmt), val)

def load_object(filename, transpose=False):
    match Path(filename).suffix:
        case '.npy': X = torch.from_numpy(np.load(filename))
        case '.pt': X = torch.load(filename)
    return X.T if transpose else X

def save_object(filename, obj):
    match Path(filename).suffix:
        case '.npy': np.save(filename, obj.numpy())
        case '.pt': torch.save(obj, filename)

def cli():
    """A performance-oriented implementation of FastICA"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')
    all_parsers = [parser]

    # whiten subcommand whitens data
    whiten_parser = subparsers.add_parser('whiten')
    all_parsers.append(whiten_parser)
    f = whiten_parser.add_argument
    f('data', type=Path)
    f('-X', '--X1-file', type=Path, default='X1.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-M', '--X-mean-file', type=Path, default='X_mean.npy')
    f('-c', '--n-component', type=int, default=0)
    f('-t', '--component-thresh', type=float, default=1e-6)
    f('--transpose', action='store_true')

    # run subcommand runs the fastica estimation algorithm
    run_parser = subparsers.add_parser('run')
    all_parsers.append(run_parser)
    f = run_parser.add_argument
    f('data', type=Path)
    f('-W', '--W-file', type=Path, default='W.npy')
    f('--w-init', type=Path)
    f('--max-iter', type=int, default=200)
    f('--tol', type=float, default=1e-4)
    f('--checkpoint-iter', type=int, default=0)
    f('--checkpoint-dir', type=Path, default='./checkpoints')
    f('--checkpoint-iter-format', type=str, default='d')
    f('--start-iter', type=int, default=0)
    f('--cwork', type=int, default=1)
    f('--retry', type=int, default=1)

    # recover subcommand produces the S & A which correspond to the X data used
    # in FastICA estimation
    recover_parser = subparsers.add_parser('recover')
    all_parsers.append(recover_parser)
    f = recover_parser.add_argument
    f('-X', '--X1-file', type=Path, default='X1.npy')
    f('-W', '--W-file', type=Path, default='W.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-S', '--S-file', type=Path, default='S.npy')
    f('-A', '--A-file', type=Path, default='A.npy')
    # u for unit scaling
    f('-u', '--scale', action='store_true')
    f('-a', '--alpha', type=float, default=1.0)
    # p for "positive" (sign flipping of major components)
    f('-p', '--sign-flip', action='store_true')
    f('-f', '--factors-file', type=Path, default='factors.npy')

    # project subcommand projects new data Y through the model defined by W, K,
    # and X_mean, with optional additional scaling factors
    project_parser = subparsers.add_parser('project')
    all_parsers.append(project_parser)
    f = project_parser.add_argument
    f('-Y', '--Y-file', type=Path, default='Y.npy')
    f('-W', '--W-file', type=Path, default='W.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-M', '--X-mean-file', type=Path, default='X_mean.npy')
    f('-f', '--factors-file', type=Path, default='factors.npy')
    f('-S', '--S-file', type=Path, default='S.npy')

    # Common arguments: stupid argparse is set up such that arguments added to
    # the parent parsers apparently don't propagate down to subcommand parsers
    for p in all_parsers:
        f = p.add_argument
        f('-v', '--verbose', action='count', default=0)
        f('--log-file', type=Path, default=None)
        f('--log-timing-format', type=str, default='.4f')
        f('--dtype', type=str, default='float64')
    del f
    args = parser.parse_args()

    match args.dtype.lower():
        case 'float'|'float32'|'f32': dtype = torch.float32
        case 'double'|'float64'|'f64': dtype = torch.float64
        case _: dtype = torch.float64
    torch.set_default_dtype(dtype)

    if args.verbose > 0 or args.log_file:
        level = logging.INFO
        if args.verbose > 1:
            level = logging.DEBUG
        logging.basicConfig(filename=args.log_file,
                            level=level,
                            format='%(asctime)s %(name)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info('FastICA %s (pid=%d): %s with parameter namespace = ',
                 __version__, os.getpid(), args.command)
    pprint_param_ns(args)

    if args.command == 'whiten':
        logging.info('Loading data file %s with%s transpose',
                     args.data, '' if args.transpose else 'out')
        XT = load_object(args.data, args.transpose)
        # whiten handles logging start/stop/etc internally
        X1, K, X_mean = learn_whitening(XT, n_component=args.n_component,
                                        component_thresh=args.component_thresh,
                                        apply=True, out=None, eps=eps_(XT))
        logging.info('Saving X1 to %s', args.X1_file)
        save_object(args.X1_file, X1)
        logging.info('Saving K to %s', args.K_file)
        save_object(args.K_file, K)
        logging.info('Saving X_mean to %s', args.X_mean_file)
        save_object(args.X_mean_file, X_mean)
        logging.info('FastICA whiten DONE')

    elif args.command == 'run':
        logging.info('Loading X1 from %s', args.data)
        X1 = load_object(args.data)
        n = X1.shape[0]
        for n_try in range(1, args.retry+1):
            logging.info('FastICA attempt %d: generating W_init', n_try)
            W_init = torch.rand(n, n)
            W, it = fastica(X1, W_init,
                            max_iter=args.max_iter,
                            tol=args.tol,
                            checkpoint_iter=args.checkpoint_iter,
                            checkpoint_dir=str(args.checkpoint_dir),
                            checkpoint_iter_format=args.checkpoint_iter_format,
                            start_iter=args.start_iter,
                            c=args.cwork,
                            log_timing_format=args.log_timing_format)
            if it < args.max_iter:
                logging.info('Attempt %d success in %d iterations', n_try, it)
                break
            # If we need to retry the algorithm, remove any checkpoints
            if args.checkpoint_iter > 0:
                shutil.rmtree(args.checkpoint_dir)
        else:
            logging.info('No convergence in %d attempts', args.retry)
        logging.info('Persisting W_init to %s', args.w_init)
        save_object(args.w_init, W_init)
        logging.info('Persisting W to %s', args.W_file)
        save_object(args.W_file, W)
        logging.info('FastICA main algorithm DONE')

    elif args.command == 'recover':
        logging.info('Loading X1 data from %s', args.X1_file)
        X1 = load_object(args.X1_file)
        logging.info('Loading est. W from %s', args.W_file)
        W = load_object(args.W_file)
        logging.info('Loading K from %s', args.K_file)
        K = load_object(args.K_file)
        logging.info('Recovering sources S')
        S = recover_S_from_WX1(W, X1)
        logging.info('Recovering components A')
        A = recover_A_from_WK(W, K)
        if args.scale:
            logging.info('Scaling A & S')
            _, _, factors = scale_to_unit_variance(S, A.T, alpha=args.alpha,
                                                   sign_flip=args.sign_flip,
                                                   inplace=True)
            logging.info('Saving scale factors to %s', args.factors_file)
            save_object(args.factors_file, factors)
        logging.info('Persisting S to %s', args.S_file)
        save_object(args.S_file, S)
        logging.info('Persisting A to %s', args.A_file)
        save_object(args.A_file, A)
        logging.info('FastICA recovering A and S DONE')

    elif args.command == 'project':
        logging.info('Loading Y data from %s', args.Y_file)
        Y = load_object(args.Y_file)
        logging.info('Loading est. W from %s', args.W_file)
        W = load_object(args.W_file)
        logging.info('Loading K from %s', args.K_file)
        K = load_object(args.K_file)
        logging.info('Loading X_mean from %s', args.X_mean_file)
        X_mean = load_object(args.X_mean_file)
        factors = None
        if args.factors_file is not None:
            logging.info('Loading optional factors from %s', args.factors_file)
            factors = load_object(args.factors_file)
        logging.info('Projecting new data Y through ICA Model')
        SY = apply_model(W, K, Y, X_mean, factors)
        logging.info('Persisting projections S(Y) to %s', args.S_file)
        save_object(args.S_file, SY)
        logging.info('Projecting new data DONE')

    else:
        parser.print_usage()
