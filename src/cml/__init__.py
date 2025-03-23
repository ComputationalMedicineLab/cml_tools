__version__ = '0.0.1'

# This appears to prevents a very bizarre "illegal instruction" error that
# seems to happen whenever the first call to `torch.tanh` in a given process is
# on a sufficiently large tensor. By putting this here I make sure that the
# first call to `torch.tanh` is always before any downstream code can run it on
# a large tensor. See issue ref:
# https://github.com/pytorch/pytorch/issues/149156#issue-2918418704
import torch
torch.tanh(torch.tensor(0))
