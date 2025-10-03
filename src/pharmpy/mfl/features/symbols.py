from typing import Literal, TypeVar

from pharmpy.internals.immutable import Immutable

T = TypeVar('T', str, Literal[''])


class Symbol(Immutable):
    pass


class Ref(Symbol):
    def __init__(self, name):
        self.name = name.upper()

    def __repr__(self):
        return f'@{self.name}'

    def __eq__(self, other):
        if not isinstance(other, Ref):
            return NotImplemented
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
