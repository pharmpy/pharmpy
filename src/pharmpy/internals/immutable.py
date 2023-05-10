from abc import ABC
from typing import Iterator, Mapping, TypeVar


class Immutable(ABC):
    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self


K = TypeVar('K')
V = TypeVar('V')


class frozenmapping(Mapping[K, V]):
    def __init__(self, mapping):
        self._mapping = mapping
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
