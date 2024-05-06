from abc import ABC
from collections.abc import Mapping
from typing import Iterator, TypeVar


def cache_method(func):
    if func.__name__ == '__hash__':

        def wrapper(self):
            if hasattr(self, '_hash'):
                return self._hash
            else:
                h = func(self)
                self._hash = h
                return h

        return wrapper
    else:
        return func


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
