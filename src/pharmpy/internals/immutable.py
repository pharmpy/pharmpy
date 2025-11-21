from abc import ABC
from collections.abc import Mapping
from functools import wraps
from typing import Iterator, TypeVar
from weakref import WeakKeyDictionary, WeakValueDictionary


def cache_method_no_args(method):
    # NOTE: This is needed because, as suprising as it sounds, no "official" solution exists.
    # SEE: https://bugs.python.org/issue45588.
    # NOTE: Adapted from https://stackoverflow.com/a/77097026.
    # NOTE: Do not try to iterate over those as it will cause issues.
    # SEE: https://stackoverflow.com/a/77100606.
    _instances = WeakValueDictionary()
    _cache = WeakKeyDictionary()

    @wraps(method)
    def wrapped(self):
        key = IdKey(self)
        try:
            return _cache[key]
        except KeyError:
            # NOTE: This prevents `key` from being GCed until `self` is GCed.
            _instances[key] = self

            # NOTE: This entry can be GCed as soon as `self` is GCed.
            value = method(self)
            _cache[key] = value
            return value

    return wrapped


class IdKey:
    # NOTE: This is needed because, as suprising as it sounds, no "official" solution exists.
    # SEE: https://bugs.python.org/issue44140.
    # SEE: https://github.com/python/cpython/issues/88306.

    def __init__(self, value):
        self._id = id(value)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id == other._id

    def __repr__(self):
        return f"<IdKey(_id={self._id})>"


class Immutable(ABC):
    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self


K = TypeVar('K')
V = TypeVar('V')


class frozenmapping(Mapping[K, V]):
    def __init__(self, mapping):
        # Do not copy if already a frozenmapping
        if isinstance(mapping, frozenmapping):
            self._mapping = mapping._mapping
            self._hash = mapping._hash
        else:
            self._mapping = dict(mapping)
            self._hash = None

    def __getitem__(self, key: K) -> V:
        return self._mapping[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self):
        return repr(self._mapping)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(tuple((k, v) for k, v in self._mapping.items()))
        return self._hash

    def replace(self, key, value):
        new = dict(self._mapping)
        new[key] = value
        return frozenmapping(new)
