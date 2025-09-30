from .mutex_feature import MutexFeature

DIRECT_EFFECT_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'))


class DirectEffect(MutexFeature):
    @classmethod
    def create(cls, type):
        type = cls._canonicalize_type(type, DIRECT_EFFECT_TYPES)
        return cls(type=type)

    def get_complexity(self):
        order = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2, 'STEP': 3, 'LOGLIN': 4}
        return order[self.type]
