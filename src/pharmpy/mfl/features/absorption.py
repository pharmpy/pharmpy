from .mutex_feature import MutexFeature

ABSORPTION_TYPES = frozenset(('FO', 'ZO', 'SEQ-ZO-FO', 'WEIBULL'))


class Absorption(MutexFeature):
    @classmethod
    def create(cls, type):
        type = cls._canonicalize_type(type, ABSORPTION_TYPES)
        return cls(type=type)

    def get_complexity(self):
        order = {'FO': 0, 'ZO': 1, 'SEQ-ZO-FO': 2, 'WEIBULL': 3}
        return order[self.type]
