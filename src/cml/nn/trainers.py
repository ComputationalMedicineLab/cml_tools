"""Pytorch implemented training tools for Pytorch Neural Nets"""
import logging
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GaussianTrainer:
    """Class for training models which predict mean and variance"""
    def __init__(self, model, optimizer, train_loader, test_loader,
                 loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        # If loss_fn is `"model"` then the model is called with each batch
        # produced by train_loader / test_loader and is expected to return a
        # triplet (losses, means, variances).
        if loss_fn is None:
            loss_fn = nn.GaussianNLLLoss(reduction='none')
        self.loss_fn = loss_fn
        # Preferred logging options (put somewhere more useful...)
        #       format='%(asctime)s %(name)s %(message)s',
        #       datefmt='%m/%d/%Y %I:%M:%S %p'
        self.log = logging.getLogger(self.__class__.__name__)
        # Log the current train / eval loss every `interval` batches; i.e.
        # sub-epochal logging.
        # TODO: actually implement the logging
        self.log_interval_train = 50
        self.log_interval_test = 50

    def run_batch(self, batch):
        if self.loss_fn == 'model':
            loss, means, variances = self.model(*batch)
        else:
            X, y = batch
            means, variances = self.model(X)
            loss = self.loss_fn(means, y, variances)
        return loss, means, variances

    def epoch_train_iter(self, epoch, loader=None):
        loader = loader or self.train_loader
        self.model.train()
        for batch_num, batch in enumerate(loader):
            loss, means, variances = self.run_batch(batch)
            yield batch_num, loss.detach(), means.detach(), variances.detach()
            # Backpropagation - XXX I assume the loss is all finite; if not, we
            # never execute this half of the function (the caller handles)
            loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def epoch_test_iter(self, epoch, loader=None):
        loader = loader or self.test_loader
        self.model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(loader):
                loss, means, variances = self.run_batch(batch)
                # No need to detach, since we are here in no_grad()
                yield batch_num, loss, means, variances

    def accumulate_epoch(self, epoch, batch_func):
        """
        Runs over all batches produced by `batch_func` and accumulates the
        results. Returns (losses, predicted means, predicted variances) as
        torch tensors concatenated over the whole output of batch_func.
        """
        total_losses = []
        total_means = []
        total_variances = []

        for batch_num, losses, means, variances in batch_func(epoch):
            total_losses.append(losses)
            total_means.append(means)
            total_variances.append(variances)
            if not torch.all(torch.isfinite(losses)):
                # XXX: Handle Inf values somehow; more than break. But breaking
                # here is important: it stops backprop if batch_func is
                # epoch_train_iter (b/c we don't re-enter the generator), so it
                # (hopefully) prevents us from ruining the model.
                break

        total_losses = torch.concatenate(total_losses)
        total_means = torch.concatenate(total_means)
        total_variances = torch.concatenate(total_variances)
        return total_losses, total_means, total_variances

    def run_epoch(self, epoch):
        """
        Runs the Training and Testing halves of an epoch and returns the
        accumulated losses, predictions, and prediction variances.
        """
        (
            train_losses,
            train_means,
            train_variances
        ) = self.accumulate_epoch(epoch, self.epoch_train_iter)
        (
            test_losses,
            test_means,
            test_variances
        ) = self.accumulate_epoch(epoch, self.epoch_test_iter)
        return (train_losses, train_means, train_variances,
                test_losses, test_means, test_variances)

    def log_inf(self, epoch, where):
        """Emit a message when a nonfinite value is detected"""
        self.log.info("Non-finite value during epoch %d %s", epoch, where)

    def train_epochs(self, start_epoch=0, n_epochs=20, results=None, keep=True,
                     collate_results=True, log_every=1, log_prefix='',
                     log_tight=False):
        """Train the model from `start_epoch` to `start_epoch+n_epochs`.

        If results is not None and not empty, then we resume the training
        using the information from the last entry of "results" (which is
        assumed to be ordered such that most recent results are at
        results[-1]).

        If `keep` is `True`, or `1`, or `"every"`, then we persist the model
        and optimizer state every epoch. If `keep` is a positive integer then
        we persist the model and optimizer state whenever `epoch % keep == 0`.
        The best model found so far is always persisted.
        """
        if (results is None) or (len(results) == 0):
            results = []
            best_loss = np.inf
            best_epoch = -1
            best_model = None
            best_optim = None
        else:
            start_epoch = results[-1]['epoch']
            best_loss = min(results[-1]['best_loss'])
            best_epoch = results[-1]['best_epoch']
            best_model = results[-1]['best_model']
            best_optim = results[-1]['best_optim']

        # If log_every is anything other than a positive number, no logging
        if not isinstance(log_every, int) or log_every < 1:
            log_every = False
        else:
            log_every = int(log_every)

        # NB the space between '%' and '{width}.{precision}f' is part of the
        # format specification, it means: leave a space for a possible negative
        # sign. The width specifier is the *total* field width, including sign,
        # decimal place, etc. The below spec is designed to ensure alignment
        # for human reading when log_tight=False.
        if log_tight:
            loss_msg = f'{log_prefix}|%d|%.6f|%.6f'
        else:
            loss_msg = '%6d | % 15.6f | % 15.6f'
            if log_prefix:
                loss_msg = f'{log_prefix} | {loss_msg}'

        for epoch in range(start_epoch, start_epoch+n_epochs):
            (
                train_losses, train_means, train_variances,
                test_losses, test_means, test_variances
            ) = self.run_epoch(epoch)
            # Log the average train and test loss
            train_loss_avg = torch.mean(train_losses).item()
            test_loss_avg = torch.mean(test_losses).item()
            if log_every and epoch % log_every == 0:
                self.log.info(loss_msg, epoch, train_loss_avg, test_loss_avg)
            # Check for whether the "best" model/optim needs to be updated. We
            # always keep a deepcopy of the best model found so far,
            # independently of what the value of `keep` is (`keep` used for
            # regular checkpointing with no regard for model performance).
            if test_loss_avg < best_loss:
                best_loss = test_loss_avg
                best_epoch = epoch
                best_model = deepcopy(self.model.state_dict())
                best_optim = deepcopy(self.optimizer.state_dict())
            res = dict(
                epoch=epoch,
                train_loss_avg=train_loss_avg,
                train_losses=train_losses,
                train_means=train_means,
                train_variances=train_variances,
                test_loss_avg=test_loss_avg,
                test_losses=test_losses,
                test_means=test_means,
                test_variances=test_variances,
                best_loss=best_loss,
                best_epoch=best_epoch,
                best_model=best_model,
                best_optim=best_optim,
                loss_fn=self.loss_fn,
            )
            # Checkpoint, if we're checkpointing every iteration...
            if keep in (True, 1, 'every'):
                res['current_model'] = deepcopy(self.model.state_dict())
                res['current_optim'] = deepcopy(self.optimizer.state_dict())
            # Or if we're checkpointing every `keep` iterations
            elif isinstance(keep, int) and (keep > 1) and (epoch % keep == 0):
                res['current_model'] = deepcopy(self.model.state_dict())
                res['current_optim'] = deepcopy(self.optimizer.state_dict())
            results.append(res)
            # If the loss (train or test) is no longer finite (either NaN or
            # pos/neg Inf), log the error and stop training.
            if not np.isfinite(train_loss_avg):
                self.log_inf(epoch, "train")
                break
            if not np.isfinite(test_loss_avg):
                self.log_inf(epoch, "train")
                break
        # Log the final epoch of stats after all epochs trained
        if log_every:
            self.log.info(loss_msg, epoch, train_loss_avg, test_loss_avg)
        if collate_results:
            collation = {}
            # Collate the tensor-valued fields
            for key in ('train_losses', 'train_means', 'train_variances',
                        'test_losses', 'test_means', 'test_variances'):
                val = [res[key] for res in results]
                collation[key] = torch.stack(val).squeeze()
            # Collate the two float-valued fields (make tensors)
            for key in ('train_loss_avg', 'test_loss_avg'):
                val = [res[key] for res in results]
                collation[key] = torch.tensor(val)
            return results, collation
        return results

    def train_epochs_simple(self, n_epochs=100, pretrain=0, convert=None):
        """
        Very simply run the model for `n_epochs` and accumulate the results.
        Optionally, train for `pretrain` epochs before beginning to accumulate
        results (total training epochs are then `n_epochs` + `pretrain`).

        Don't do any checkpointing, logging, etc. Primary use case is for
        overfitting self.model to some small dataloaders, where we only want
        the results and expect them quickly.
        """
        for i in range(pretrain):
            self.run_epoch(i)
        results = [list() for _ in range(6)]
        for i in range(n_epochs):
            for acc, out in zip(results, self.run_epoch(i)):
                acc.append(out)
        if callable(convert):
            return [convert(x) for x in results]
        return results
