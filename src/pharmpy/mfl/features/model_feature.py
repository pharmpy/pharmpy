import builtins
from abc import abstractmethod

from pharmpy.internals.immutable import Immutable


class ModelFeature(Immutable):
    @property
    @abstractmethod
    def args(self):
        pass

    @staticmethod
    @abstractmethod
    def repr_many(features):
        pass

    @staticmethod
    def _canonicalize_type(type, types, name=None):
        if not name:
            name = 'type'
        if not isinstance(type, str):
            raise TypeError(f'Type of `{name}` must be a string: got {builtins.type(type)}')
        type = type.upper()
        if type.upper() not in types:
            raise ValueError(f'Unknown `{name}`: got {type}')
        return type.upper()
