from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Self, TypeVar

from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures

T = TypeVar('T')


class MutexFeature(ModelFeature):
    def __init__(self, type: str):
        self._type = type

    @classmethod
    @abstractmethod
    def create(cls, type: str) -> Self:
        pass

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        return self.__class__.create(type=type)

    @property
    def type(self) -> str:
        return self._type

    @property
    def args(self) -> tuple[str]:
        return (self.type,)

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        return self.type == other.type

    def __lt__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            return False

        return self.get_complexity() < other.get_complexity()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__.upper()}({self.type})'

    @abstractmethod
    def get_complexity(self) -> int:
        pass

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, MutexFeature))
        assert len(features) == len(mf.features)
        if len(features) == 1:
            return repr(features[0])
        class_name = features[0].__class__.__name__.upper()
        return f"{class_name}([{','.join(feat.type for feat in features)}])"
