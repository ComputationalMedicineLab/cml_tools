"""
Snippets of code and associated notes/documentation for coding idioms, sharp
corners, and other things we wish to document and remember but don't have a
need for currently in the main codebase.
"""
# Locks are hard: https://stackoverflow.com/a/69913167
# This is apparently an issue with the Pool rather than passing locks as
# arguments. Locks passed as arguments to Process objects are correct; locks
# passed as arguments to Pool workers are incorrect (bc the pool behind the
# scenes is using a Queue to coordinate its workers, and the lock cannot
# survive pickling through the queue).
def init_proc(_lock):
    """Put a Lock into a subprocess's global namespace, for use with Pools:

    >>> from multiprocessing import Lock
    >>> from multiprocessing.pool import Pool
    >>> with Pool(N_PROC, initializer=init_proc, initargs=(Lock(),)) as pool:
    ...     # do stuff with worker functions that synchronize on the lock
    """
    global lock
    lock = _lock


# How does one get the index of an element in a numpy ndarray *quickly*?
def np_index_of(ndarr, item):
    """This function is basically documentation; it should always be inlined.

    >>> # Timings in ipython:
    >>> # x is 400_000 random vec, `n` a random element, xl = x.tolist()
    In [45]: %timeit xl.index(n)
        ...: %timeit np.argmax(x == n)
        ...: %timeit np.argmin(x != n)
        ...: %timeit np.where(x == n)[0][0]
        ...: %timeit np.argwhere(x == n)[0][0]
        ...: %timeit np.nonzero(x == n)[0][0]
        ...:
    701 μs ± 164 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    154 μs ± 1.11 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    172 μs ± 67.1 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    208 μs ± 137 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    213 μs ± 146 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    207 μs ± 80 ns per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    """
    return np.argmax(ndarr == item)
