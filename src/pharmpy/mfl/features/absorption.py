from __future__ import annotations

from .mutex_feature import MutexFeature

ABSORPTION_TYPES = frozenset(('FO', 'ZO', 'SEQ-ZO-FO', 'WEIBULL'))


class Absorption(MutexFeature):
    _COMPLEXITY_ORDER = {'FO': 0, 'ZO': 1, 'SEQ-ZO-FO': 2, 'WEIBULL': 3}

    @classmethod
    def create(cls, type: str) -> Absorption:
        type = cls._canonicalize_type(type, ABSORPTION_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
