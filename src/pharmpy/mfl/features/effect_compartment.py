from __future__ import annotations

from .mutex_feature import MutexFeature

EFFECT_COMP_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'))


class EffectComp(MutexFeature):
    _COMPLEXITY_ORDER = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2, 'STEP': 3, 'LOGLIN': 4}

    @classmethod
    def create(cls, type: str) -> EffectComp:
        type = cls._canonicalize_type(type, EFFECT_COMP_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
