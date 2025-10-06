from __future__ import annotations

from .mutex_feature import MutexFeature

METABOLITE_TYPES = frozenset(('PSC', 'BASIC'))


class Metabolite(MutexFeature):
    _COMPLEXITY_ORDER = {'PSC': 0, 'BASIC': 1}

    @classmethod
    def create(cls, type: str) -> Metabolite:
        type = cls._canonicalize_type(type, METABOLITE_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
