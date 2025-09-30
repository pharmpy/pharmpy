from .help_functions import get_repr, group_args
from .model_feature import ModelFeature

INDIRECT_EFFECT_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID'))
PRODUCTION_TYPES = frozenset(('DEGRADATION', 'PRODUCTION'))


class IndirectEffect(ModelFeature):
    def __init__(self, type, production_type):
        self._type = type
        self._production_type = production_type

    @classmethod
    def create(cls, type, production_type):
        type = cls._canonicalize_type(type, INDIRECT_EFFECT_TYPES)
        production_type = cls._canonicalize_type(
            production_type, PRODUCTION_TYPES, 'production_type'
        )
        return cls(type=type, production_type=production_type)

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        production_type = kwargs.get('production_type', self.production_type)
        return IndirectEffect.create(type=type, production_type=production_type)

    @property
    def type(self):
        return self._type

    @property
    def production_type(self):
        return self._production_type

    @property
    def args(self):
        return self.type, self.production_type

    def __repr__(self):
        return f'INDIRECTEFFECT({self.type},{self.production_type})'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, IndirectEffect):
            return False
        return self.type == other.type and self.production_type == other.production_type

    def __lt__(self, other):
        if not isinstance(other, IndirectEffect):
            return NotImplemented
        if self == other:
            return False
        if self.production_type != other.production_type:
            type_rank = {'DEGRADATION': 0, 'PRODUCTION': 1}
            return type_rank[self.production_type] < type_rank[other.production_type]
        type_rank = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2}
        return type_rank[self.type] < type_rank[other.type]

    @staticmethod
    def repr_many(features):
        features = sorted(features)
        no_of_args = len(features[0].args)

        args_grouped = group_args([feature.args for feature in features], i=no_of_args)

        indirect_effect_repr = []
        for arg in args_grouped:
            type, production_type = arg
            inner = f'{get_repr(type)},{get_repr(production_type)}'
            indirect_effect_repr.append(f'INDIRECTEFFECT({inner})')

        return ';'.join(indirect_effect_repr)
