from __future__ import annotations

from pharmpy.internals.immutable import frozenmapping

from .mutex_feature import MutexFeature

ELIMINATION_TYPES = frozenset(('FO', 'ZO', 'MM', 'MIX-FO-MM'))


class Elimination(MutexFeature):
    _COMPLEXITY_ORDER = frozenmapping({'FO': 0, 'ZO': 1, 'MM': 2, 'MIX-FO-MM': 3})

    @classmethod
    def create(cls, type: str) -> Elimination:
        type = cls._canonicalize_type(type, ELIMINATION_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
