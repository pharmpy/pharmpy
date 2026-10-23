# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import annotations

from pharmpy.internals.immutable import frozenmapping

from .mutex_feature import MutexFeature

METABOLITE_TYPES = frozenset(('PSC', 'BASIC'))


class Metabolite(MutexFeature):
    _COMPLEXITY_ORDER = frozenmapping({'PSC': 0, 'BASIC': 1})

    @classmethod
    def create(cls, type: str) -> Metabolite:
        type = cls._canonicalize_type(type, METABOLITE_TYPES)
        return cls(type=type)

    def get_complexity(self) -> int:
        return self._COMPLEXITY_ORDER[self.type]
