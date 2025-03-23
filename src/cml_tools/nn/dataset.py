"""Functions and classes for dealing with Datasets as ndarrays or tensors"""
import collections
import itertools
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset


class BatchSampler(Sampler):
    def __init__(self, n, batch_size=32, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(n))

    def __len__(self):
        n_batch, n_rem = divmod(len(self.indices), self.batch_size)
        if n_rem > 0 and not self.drop_last:
            n_batch += 1
        return n_batch

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for index_set in itertools.batched(self.indices, n=self.batch_size):
            index_set = list(index_set)
            if len(index_set) == self.batch_size:
                yield index_set
        if not self.drop_last:
            yield index_set


class BatchLoader:
    # Significantly faster than a pytorch DataLoader (20%-30% the wall time)
    # XXX: if this makes it into serious usage (w. GPU) may need to implement
    # memory pinning and prefetching. Or, just use for exploratory work.
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.sampler = BatchSampler(len(dataset), batch_size=batch_size,
                                    shuffle=shuffle, drop_last=drop_last)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        for indices in self.sampler:
            yield self.dataset[indices]


class InstanceDS(Dataset):
    """Dataset class for pairing instances and targets"""
    def __init__(self, instances, targets, convert=True):
        assert len(instances) == len(targets)
        if len(targets.shape) < 2:
            targets = targets.reshape(-1, 1)
        if convert and isinstance(instances, np.ndarray):
            instances = torch.from_numpy(instances).to(torch.float32)
        if convert and isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(torch.float32)
        self.instances = instances
        self.targets = targets

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index], self.targets[index]

    @property
    def shape(self):
        return self.instances.shape

    @classmethod
    def from_npz(cls, filename, X_key, y_key):
        data = np.load(filename)
        return cls(data[X_key], data[y_key])

    @classmethod
    def train_test_from_npz(cls, filename):
        """Return train / test InstanceDS objects from an .npz"""
        data = np.load(filename)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        return cls(X_train, y_train), cls(X_test, y_test)

    def loader(self, batch_size=32, shuffle=True, drop_last=False):
        """Return a BatchLoader for this Dataset object"""
        return BatchLoader(self, batch_size, shuffle, drop_last)

    def sample(self, fraction=0.2, indices=None):
        """Return a SubSampleDS of this Dataset"""
        # argument `indices` takes priority over fraction
        if indices is not None:
            return SubSampleDS(self, indices)
        elif fraction is not None:
            indices = np.random.randint(0, len(self), int(fraction*len(self)))
            return SubSampleDS(self, indices)
        else:
            return self


class SubSampleDS(Subset):
    """Subclass of torch.utils.data.Subset intended for InstanceDS Subsets"""
    @property
    def instances(self):
        return self.dataset.instances[self.indices]

    @property
    def targets(self):
        return self.dataset.targets[self.indices]

    def loader(self, batch_size=32, shuffle=True, drop_last=False):
        """Return a BatchLoader for this Dataset object"""
        return BatchLoader(self, batch_size, shuffle, drop_last)

    def resample(self, fraction=0.2):
        return self.dataset.get_subset(fraction=fraction)


class RepeaterDS(InstanceDS):
    """Indexes its instances, targets pair `n_repeat` times in sequence"""
    def __init__(self, instances, targets, convert=True, n_repeat=1000):
        super().__init__(instances, targets, convert)
        self.n_repeat = n_repeat

    def __len__(self):
        return super().__len__() * self.n_repeat

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index
        if index >= len(self):
            raise IndexError('list index out of range')
        index = index % len(self.instances)
        return super().__getitem__(index)

    def loader(self, batch_size=None, shuffle=False, drop_last=False):
        """Return a BatchLoader for this Dataset object"""
        batch_size = batch_size or len(self.instances)
        return BatchLoader(self, batch_size, shuffle, drop_last)


def save_to_npz(file, datasets: dict[str, InstanceDS]):
    """Persists a mapping of datasets as an .npz"""
    to_save = {}
    for key, ds in datasets.items():
        if isinstance(ds.instances, torch.Tensor):
            to_save[f'X_{key}'] = ds.instances.to(torch.float64).numpy()
            to_save[f'y_{key}'] = ds.targets.to(torch.float64).numpy()
        else:
            to_save[f'X_{key}'] = ds.instances
            to_save[f'y_{key}'] = ds.targets
    np.savez(file, **to_save)


def load_from_npz(file, ds_class=InstanceDS):
    """Load a dictionary of datasets from an .npz"""
    # Very tolerantly ignores any keys which don't form X_*, y_* pairs
    data = np.load(file)
    keys = [s[2:] for s in data if s.startswith('X_')]
    ds = {}
    for k in keys:
        X_key = f'X_{k}'
        y_key = f'y_{k}'
        if X_key in data and y_key in data:
            ds[k] = ds_class(data[X_key], data[y_key])
    return ds
