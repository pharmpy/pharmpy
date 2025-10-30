from __future__ import annotations

import builtins
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Self, Sequence

from pharmpy.internals.immutable import Immutable

from .symbols import Ref

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class ModelFeature(Immutable):
    @abstractmethod
    def replace(self, **kwargs) -> Self:
        pass

    @property
    @abstractmethod
    def args(self) -> tuple[Any, ...]:
        pass

    @staticmethod
    @abstractmethod
    def repr_many(mf: ModelFeatures) -> str:
        pass

    def expand(self, expand_to: Mapping[Ref, Sequence[str]]) -> tuple[ModelFeature, ...]:
        raise NotImplementedError

    def is_expanded(self):
        arg_types = (type(arg) for arg in self.args)
        if Ref in arg_types:
            return False
        else:
            return True

    @staticmethod
    def _canonicalize_type(type: str, types: Iterable[str], name: Optional[str] = None) -> str:
        if not name:
            name = 'type'
        if not isinstance(type, str):
            raise TypeError(f'Type of `{name}` must be a string: got {builtins.type(type)}')
        type = type.upper()
        if type.upper() not in types:
            raise ValueError(f'Unknown `{name}`: got {type}')
        return type.upper()
