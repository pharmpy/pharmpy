from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

T = TypeVar('T', str, Literal[''])


class Symbol:
    pass


@dataclass(frozen=True)
class Name(Symbol, Generic[T]):
    name: T


@dataclass(frozen=True)
class Wildcard(Symbol):
    pass
