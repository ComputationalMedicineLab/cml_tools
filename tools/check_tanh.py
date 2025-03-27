#!/bin/env python
# Under some odd circumstances (very likely the same circumstances that cause
# the segfaults instead of IndexError in ./check_segfault.py) running
# torch.tanh on a large array will produce an "illegal hardware exception" - I
# think this happens when the first call is large enough to dispatch to a
# multi-core implementation backing tanh, and something in the code is
# uninitialized. gh ref:
# https://github.com/pytorch/pytorch/issues/149156#issue-2918418704
import os
print('LD_PRELOAD: ', os.getenv('LD_PRELOAD', 'is unset'))
import torch
X = torch.rand(2_000, 600_000)
torch.tanh(X)
