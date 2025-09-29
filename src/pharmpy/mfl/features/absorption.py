from .mutex_feature import MutexFeature

ABSORPTION_TYPES = frozenset(('FO', 'ZO', 'SEQ-ZO-FO', 'WEIBULL'))


class Absorption(MutexFeature):
    order = {'FO': 0, 'ZO': 1, 'SEQ-ZO-FO': 2, 'WEIBULL': 3}

    @classmethod
    def create(cls, type):
        super().create(type)
        type = type.upper()
        if type not in ABSORPTION_TYPES:
            raise ValueError(f'Unknown `type`: got {type}')
        return cls(type=type)
