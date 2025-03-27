#!/bin/env python
# Under some odd circumstances this will cause a segfault instead of the
# intended IndexError (if LD_PRELOAD loads an ABI-incompatible version of iomp,
# for example). This script checks for that behavior. gh ref:
# https://github.com/pytorch/pytorch/issues/149157#issue-2918446237
import os
print('LD_PRELOAD: ', os.getenv('LD_PRELOAD', 'is unset'))
import torch
torch.rand(100)[101]
