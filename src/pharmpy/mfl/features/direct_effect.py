from __future__ import annotations

from .mutex_feature import MutexFeature

DIRECT_EFFECT_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'))


class DirectEffect(MutexFeature):
    _COMPLEXITY_ORDER = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2, 'STEP': 3, 'LOGLIN': 4}

    @classmethod
    def create(cls, type: str) -> DirectEffect:
        type = cls._canonicalize_type(type, DIRECT_EFFECT_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
