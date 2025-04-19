import pickle
from cml import pickle_stream, unpickle_stream


class Record:
    """
    Superclass for defining generic Record classes: lightweight, fast-access
    and fast-serializing containers for fixed sets of fields with variably
    sized data types. Use numpy records and recarrays for struct like types
    that have a fixed set of fixed-width data types.

    Provides mostly conversion method boilerplate.

    Subclasses should provide `fields` and `dtypes` at the class level, set
    `__slots__` to `fields`, and provide an explicit `__init__`. No metaclass
    magic generates these things for you or protects from failure to define
    them.

    Also, explicitly defining `astuple` in the subclass is often 10x faster
    than the default generic implementation using `getattr`.

    Usage
    -----
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
    """
    fields = NotImplemented
    dtypes = NotImplemented
    __slots__ = ()

    @property
    def astuple(self):
        return tuple(getattr(self, name) for name in self.fields)

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
        with open(filename, mode) as file:
            pickle.dump(self.astuple, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, filename):
        """Read a single object tuple (cls.astuple) from a file"""
        with open(filename, 'rb') as file:
            return cls(*pickle.load(file))

    @classmethod
    def to_pickle_seq(cls, records, filename, mode='wb', header=False):
        """Write an Iterable as a single pickled Sequence to a file"""
        data = cls.to_tuples(records)
        if header:
            data.insert(0, cls.fields)
        with open(filename, mode) as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle_seq(cls, filename, header=False):
        """Read a Sequence from a single pickle in a file"""
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        if header:
            data = data[1:]
        return cls.from_iter(data, apply_types=False)

    @classmethod
    def to_pickle_stream(cls, records, filename, mode='wb', header=False):
        """Write an Iterable as a stream of pickles to a file"""
        with open(filename, mode) as file:
            if header:
                pickle.dump(cls.fields, file, protocol=pickle.HIGHEST_PROTOCOL)
            for r in records:
                pickle.dump(r.astuple, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle_stream(cls, filename, header=False):
        """Read a Sequence from a stream of pickles in a file"""
        stream = unpickle_stream(filename)
        if header: next(stream)
        return cls.from_iter(stream)
