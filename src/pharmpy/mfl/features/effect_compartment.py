from .mutex_feature import MutexFeature

EFFECT_COMP_TYPES = frozenset(('LINEAR', 'EMAX', 'SIGMOID', 'STEP', 'LOGLIN'))


class EffectComp(MutexFeature):
    @classmethod
    def create(cls, type):
        super().create(type)
        type = type.upper()
        if type not in EFFECT_COMP_TYPES:
            raise ValueError(f'Unknown `type`: got {type}')
        return cls(type=type)

    def get_complexity(self):
        order = {'LINEAR': 0, 'EMAX': 1, 'SIGMOID': 2, 'STEP': 3, 'LOGLIN': 4}
        return order[self.type]
