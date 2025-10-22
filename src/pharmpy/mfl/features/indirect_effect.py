from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from .help_functions import get_repr, group_args
from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures

INDIRECT_EFFECT_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID'))


class IndirectEffect(ModelFeature):
    def __init__(self, type: str, production: bool):
        self._type = type
        self._production = production

    @classmethod
    def create(cls, type: str, production: bool) -> IndirectEffect:
        type = cls._canonicalize_type(type, INDIRECT_EFFECT_TYPES)
        if not isinstance(production, bool):
            raise TypeError(f'Type of `production` must be a bool: got {builtins.type(type)}')

        return cls(type=type, production=production)

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        production = kwargs.get('production', self.production)
        return IndirectEffect.create(type=type, production=production)

    @property
    def type(self) -> str:
        return self._type

    @property
    def production(self) -> bool:
        return self._production

    @property
    def args(self) -> tuple[str, bool]:
        return self.type, self.production

    def __repr__(self) -> str:
        production_type = 'PRODUCTION' if self.production else 'DEGRADATION'
        return f'INDIRECTEFFECT({self.type},{production_type})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, IndirectEffect):
            return False
        return self.type == other.type and self.production == other.production

    def __lt__(self, other) -> bool:
        if not isinstance(other, IndirectEffect):
            return NotImplemented
        if self == other:
            return False
        if self.production != other.production:
            return self.production < other.production
        type_rank = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2}
        return type_rank[self.type] < type_rank[other.type]

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, IndirectEffect))
        assert len(features) == len(mf.features)

        no_of_args = len(features[0].args)
        args_grouped = group_args([feature.args for feature in features], i=no_of_args)

        indirect_effect_repr = []
        for arg in args_grouped:
            type, production = arg
            if isinstance(production, tuple):
                production_type = ('DEGRADATION', 'PRODUCTION')
            else:
                production_type = 'PRODUCTION' if production else 'DEGRADATION'
            inner = f'{get_repr(type)},{get_repr(production_type)}'
            indirect_effect_repr.append(f'INDIRECTEFFECT({inner})')

        return ';'.join(indirect_effect_repr)
