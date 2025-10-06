from __future__ import annotations

from typing import TYPE_CHECKING

from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class LagTime(ModelFeature):
    def __init__(self, on: bool):
        self._on = on

    @classmethod
    def create(cls, on: bool) -> LagTime:
        if not isinstance(on, bool):
            raise TypeError(f'Type of `type` must be a bool: got {type(on)}')
        return cls(on=on)

    def replace(self, **kwargs):
        on = kwargs.get('on', self.on)
        return LagTime.create(on=on)

    @property
    def on(self) -> bool:
        return self._on

    @property
    def args(self) -> tuple[bool]:
        return (self.on,)

    def __repr__(self) -> str:
        inner = 'ON' if self.on else 'OFF'
        return f'LAGTIME({inner})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, LagTime):
            return False
        return self.on == other.on

    def __lt__(self, other) -> bool:
        if not isinstance(other, LagTime):
            return NotImplemented
        if self == other:
            return False
        return self.on < other.on

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, LagTime))
        assert len(features) == len(mf.features)
        if len(features) == 1:
            return repr(features[0])
        else:
            inner = f"[{','.join('ON' if feat.on else 'OFF' for feat in features)}]"
        return f'LAGTIME({inner})'
