from abc import ABC


class Immutable(ABC):
    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self
