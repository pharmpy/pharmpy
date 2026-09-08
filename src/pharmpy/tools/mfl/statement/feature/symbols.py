from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar('T', str, Literal[''])


class Symbol:
    pass


@dataclass(frozen=True)
class Name[T](Symbol):
    name: T


@dataclass(frozen=True)
class Wildcard(Symbol):
    pass


@dataclass(frozen=True)
class Option(Symbol):
    option: bool
