import builtins

from .model_feature import ModelFeature

ABSORPTION_TYPES = frozenset(('FO', 'ZO', 'SEQ-ZO-FO', 'WEIBULL'))


class Absorption(ModelFeature):
    def __init__(self, type):
        self._type = type

    @classmethod
    def create(cls, type):
        if not isinstance(type, str):
            raise TypeError(f'Type of `type` must be a string: got {builtins.type(type)}')
        type = type.upper()
        if type not in ABSORPTION_TYPES:
            raise ValueError(f'Unknown `type`: got {type}')
        return cls(type=type)

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        return Absorption.create(type=type)

    @property
    def type(self):
        return self._type

    @property
    def args(self):
        return (self.type,)

    def __repr__(self):
        return f'ABSORPTION({self.type})'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Absorption):
            return False
        return self.type == other.type

    def __lt__(self, other):
        if not isinstance(other, Absorption):
            return NotImplemented
        if self == other:
            return False
        type_rank = {'FO': 0, 'ZO': 1, 'SEQ-ZO-FO': 2, 'WEIBULL': 3}

        def _get_complexity(obj):
            return type_rank.get(obj.type)

        return _get_complexity(self) < _get_complexity(other)

    @staticmethod
    def repr_many(features):
        if len(features) == 1:
            return repr(features[0])
        return f"ABSORPTION([{','.join(feat.type for feat in features)}])"
