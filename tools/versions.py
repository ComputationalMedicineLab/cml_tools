#!/bin/env python
import os
verbose = os.getenv('VERBOSE', False)

import cython, numpy, scipy, sklearn, torch
sep = ("-"*80)

if verbose:
    print(sep)
    numpy.show_config()
    print(sep)
    print(torch.__config__.show())

print(sep)
print(f'Numpy version: {numpy.__version__}')
print(f'Cython version: {cython.__version__}')
print(f'scipy version: {scipy.__version__}')
print(f'sklearn version: {sklearn.__version__}')
print(f'torch version: {torch.__version__}')
print(sep)
