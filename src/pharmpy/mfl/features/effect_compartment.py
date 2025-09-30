from .mutex_feature import MutexFeature

EFFECT_COMP_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'))


class EffectComp(MutexFeature):
    @classmethod
    def create(cls, type):
        type = cls._canonicalize_type(type, EFFECT_COMP_TYPES)
        return cls(type=type)

    def get_complexity(self):
        order = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2, 'STEP': 3, 'LOGLIN': 4}
        return order[self.type]
