import pickle


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
    def from_iter(cls, records, apply_types=False):
        if apply_types:
            return tuple(cls.apply_types(t) for t in records)
        return tuple(cls(*t) for t in records)

    @classmethod
    def to_tuples(cls, records):
        return tuple(t.astuple for t in records)

    @classmethod
    def from_pkl(cls, filename, header=True):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        # Strip off a "header" object if present
        if header:
            data = data[1:]
        return cls.from_iter(data, apply_types=False)

    @classmethod
    def to_pkl(cls, filename, records):
        with open(filename, 'wb') as file:
            pickle.dump(cls.to_tuples(records), file,
                        protocol=pickle.HIGHEST_PROTOCOL)
