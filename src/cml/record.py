import io
import pathlib
import pickle
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from isal import igzip

from cml import unpickle_stream


def _open(filename, mode, *args, **kwargs):
    """Helper func to auto-detect if gzip is in use"""
    if '.gz' in pathlib.Path(filename).suffixes:
        opener = igzip.open
    else:
        opener = io.open
    return opener(filename, mode, *args, **kwargs)


class Record:
    """
    Superclass for defining two flavors of generic Record type: AOS (Array of
    Structs) and SOA (Struct of Arrays). In either paradigm the subclass
    defines the Struct as a lightweight, fast-access and fast-serializing
    container.

    The Array of Structs paradigm makes most sense when the fields of the
    record are expected to be arbitrary Python objects or objects of unknown
    size and shape.

    The Struct of Arrays paradigm makes most sense when the fields of the
    record are expected to have a fixed (or at least fixed-size) datatype.

    The Record superclass defines many IO and conversion methods with some
    other boilerplate. Most methods are suitable for either paradigm, some are
    more suitable for one. It is the programmers responsibility to understand
    when a method defined on the Record superclass is appropriate for use with
    any particular subclass in any particular sitation.

    Subclasses must set a class variable `fields` to a tuple of string names,
    and set the class variable `__slots__ = fields`. Optionally a subclass may
    set a tuple of types as the class variable `dtypes`. This class var is used
    in one or two conversion functions, but is otherwise just literate
    programming. Subclasses must define explicitly an `__init__` function which
    takes the field values as positional args, in the same order as the fields.
    This is expected by many of the classmethod alternate constructors defined
    below.

    Optionally, if a subclass explicitly defines the property `astuple`, the
    explicit definition is often significantly faster than the default Record
    implementation using `getattr`.

    Example:
    >>> class Point(Record):
    ...     fields = ('x', 'y', 'z')
    ...     dtypes = (float, float, float)
    ...     __slots__ = fields
    ...     def __init__(self, x, y, z):
    ...         self.x = x
    ...         self.y = y
    ...         self.z = z
    ...     # Explicit astuple is often 10x faster than Record.astuple
    ...     @property
    ...     def astuple(self):
    ...         return (self.x, self.y, self.z)
    >>> # Usage in Array-of-Structs: a list of 1000 scalar points
    >>> points = [Point(*xyz) for xyz in np.random.rand(1000, 3)]
    >>> # Usage in Struct-of-Arrays: a Point object with ndarray field values
    >>> points = Point(*np.random.rand(3, 1000))
    """
    fields = NotImplemented
    dtypes = NotImplemented
    __slots__ = ()

    @property
    def astuple(self):
        """Return the object values as a tuple in field order"""
        return tuple(getattr(self, name) for name in self.fields)

    @property
    def asdict(self):
        """Return a dict mapping object field names to values"""
        return dict(zip(self.fields, self.astuple))

    # from_iter and to_tuples are inverses of each other to be used in reading
    # and writing pickle files. Remember, never pickle our bespoke objects -
    # the pickle is very likely to long outlive our code. apply_types is
    # intended as a subroutine of from_iter when applied to, e.g. CSV files.
    @classmethod
    def apply_types(cls, rec):
        return cls(*(f(x) for f, x in zip(cls.dtypes, rec)))

    @classmethod
    def to_tuples(cls, records):
        """Strip the class from an Iterable of records"""
        return tuple(t.astuple for t in records)

    @classmethod
    def from_iter(cls, records, apply_types=False, unzip=False):
        """Apply the class to an Iterable of records"""
        if unzip:
            records = zip(*records)
        if apply_types:
            return tuple(cls.apply_types(t) for t in records)
        return tuple(cls(*t) for t in records)

    # Pickle read/write pairs:
    # (to_pickle, from_pickle) - reads/writes single objects
    # (to_pickle_seq, from_pickle_seq) - read/write single pickled sequence
    # (to_pickle_stream, etc) - read/write as sequence of pickles
    def to_pickle(self, filename, mode='wb'):
        """Write the object tuple (self.astuple) to a file"""
        with _open(filename, mode) as file:
            pickle.dump(self.astuple, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filename):
        """Read a single object tuple (cls.astuple) from a file"""
        with _open(filename, 'rb') as file:
            return cls(*pickle.load(file))

    @classmethod
    def to_pickle_seq(cls, records, filename, mode='wb', header=False):
        """Write an Iterable as a single pickled Sequence to a file"""
        data = cls.to_tuples(records)
        if header:
            data.insert(0, cls.fields)
        with _open(filename, mode) as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle_seq(cls, filename, header=False):
        """Read a Sequence from a single pickle in a file"""
        with _open(filename, 'rb') as file:
            data = pickle.load(file)
        if header:
            data = data[1:]
        return cls.from_iter(data, apply_types=False)

    @classmethod
    def to_pickle_stream(cls, records, filename, mode='wb', header=False):
        """Write an Iterable as a stream of pickles to a file"""
        with _open(filename, mode) as file:
            if header:
                pickle.dump(cls.fields, file, protocol=pickle.HIGHEST_PROTOCOL)
            for r in records:
                pickle.dump(r.astuple, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle_stream(cls, filename, header=False):
        """Read a Sequence from a stream of pickles in a file"""
        with _open(filename, 'rb') as stream:
            stream = unpickle_stream(stream)
            if header: next(stream)
            return cls.from_iter(stream)

    # Numpy read / write pairs:
    # (1) to / from npz: read and write .npz files using self.fields as keys
    # and self.astuple as the values. This really makes most sense if all the
    # data fields are ndarrays or other numpy objects; i.e. the "Record"
    # subclass is in SoA (Struct of Arrays) format rather than the AoS (Array
    # of Structs) format some of the above pickling functions expect.
    def to_npz(self, filename, compress=True):
        """Write data to an .npz NpzFile"""
        if compress:
            np.savez_compressed(filename, **self.asdict)
        else:
            np.savez(filename, **self.asdict)

    @classmethod
    def from_npz(cls, filename):
        """Read data from an .npz NpzFile"""
        return cls(**np.load(filename))

    def to_npy(self, filename, field='data'):
        """Write a single field to an .npy file"""
        np.save(filename, getattr(self, field))

    @classmethod
    def from_npy(cls, filename):
        """Read a single field from an .npy file"""
        return cls(np.load(filename))


class SOA(Record):
    """An ndarray backed Structure-of-Arrays"""
    fields = NotImplemented
    dtypes = NotImplemented
    __slots__ = ()

    @property
    def arr_nbytes(self):
        """Tuple of `nbytes` per underlying ndarray"""
        return tuple(x.nbytes for x in self.astuple)

    @property
    def arr_shapes(self):
        """Tuple of `shape` per underlying ndarray"""
        return tuple(x.shape for x in self.astuple)

    @property
    def arr_dtypes(self):
        """Tuple of `dtype` per underlying ndarray"""
        return tuple(x.dtype for x in self.astuple)

    def lexsort(self):
        """
        Sort the ndarrays underlying the instance lexically by field value in
        the field order and return a new instance wrapping the sorted arrays.
        """
        # Lexsort sorts from the *final* key backwards, which is the opposite
        # behavior of the `key` parameter to `sorted`, and so seems unintuitive.
        # Hence this wrapper function to keep straight how to sort correctly.
        return self[np.lexsort(tuple(reversed(self.astuple)))]

    def __eq__(self, other):
        """
        Returns True if the underlying ndarrays have the same shape and values.
        Treats nan values as equal to each other.
        """
        return all(np.array_equal(x, y, equal_nan=True)
                   for x, y in zip(self.astuple, other.astuple))

    def __len__(self):
        """Returns the (common) length of the underlying ndarrays."""
        return len(self.astuple[0])

    def __iter__(self):
        """Yields tuples of values from the underlying ndarrays."""
        yield from zip(*self.astuple)

    def __getitem__(self, index):
        """
        If `index` is a string and matches one of the object fields, return the
        ndarray backing that field. If an integer, return the tuple of values
        at that index. Otherwise attempt advanced ndarray indexing across each
        underlying ndarray in parallel.
        """
        match index:
            case int(index):
                return tuple(x[index] for x in self.astuple)
            case str(key):
                if key in self.fields:
                    return getattr(self, key)
                else:
                    raise KeyError(f'{key=} not found in {self}')
            case _:
                return self.__class__(*(x[index] for x in self.astuple))

    def make_shared_memory(self, smm: SharedMemoryManager):
        """
        Acquires a SharedMemory buffer from the provided SharedMemoryManager
        instance, wraps the buffer in appropriate ndarrays, copies the data
        from self into the shared memory, and returns (mem, wrapper).
        """
        mem = smm.SharedMemory(sum(self.arr_nbytes))
        wrap = self.wrap_shared_memory(self.arr_shapes, mem)
        for field in self.fields:
            src = getattr(self, field)
            dst = getattr(wrap, field)
            np.copyto(dst, src)
        return mem, wrap

    @classmethod
    def wrap_shared_memory(cls, shapes, mem: SharedMemory):
        """
        Provided a SharedMemory buffer loaded with data and a set of shapes,
        wraps the shared memory in appropriately typed ndarrays and returns the
        shared-memory backed instance.
        """
        i = 0
        args = []
        for field, dtype, shape in zip(cls.fields, cls.dtypes, shapes):
            nbytes = np.multiply.reduce((dtype.itemsize, *shape))
            arr = np.ndarray(shape, dtype=dtype, buffer=mem.buf[i:i+nbytes])
            i += nbytes
            args.append(arr)
        return cls(*args)

    @classmethod
    def from_arrow(cls, table):
        """Construct from an Apache Arrow pyarrow.Table object"""
        if NotImplemented in (cls.fields, cls.dtypes):
            raise NotImplementedError
        args = []
        for field, dtype in zip(cls.fields, cls.dtypes):
            arr = table.column(field).to_numpy()
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)
            args.append(arr)
        return cls(*args)

    @classmethod
    def from_parquet(cls, filename):
        """Read from a parquet file"""
        # Fail early if we can
        if NotImplemented in (cls.fields, cls.dtypes):
            raise NotImplementedError
        import pyarrow.parquet as pq
        return cls.from_arrow(pq.read_table(filename).combine_chunks())

    @classmethod
    def from_records(cls, records, sort=True):
        """
        The inverse of __iter__: iter(self) produces tuples of values taken
        from each underlying ndarray; from_records reassembles those tuples
        into wrapped ndarrays.
        """
        if sort:
            records = sorted(records)
        args = []
        for dtype, values in zip(cls.dtypes, *zip(records)):
            args.append(np.array(values, dtype=dtype))
        return cls(*args)
