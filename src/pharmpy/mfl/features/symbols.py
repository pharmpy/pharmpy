from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar('T', str, Literal[''])


class Symbol:
    pass


@dataclass(frozen=True)
class Ref(Symbol):
    name: str

    def __repr__(self):
        return f'@{self.name.upper()}'

    def __lt__(self, other):
        if isinstance(other, str):
            return True
        else:
            return self.name < other.name
