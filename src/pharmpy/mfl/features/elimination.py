from .mutex_feature import MutexFeature

ELIMINATION_TYPES = frozenset(('FO', 'ZO', 'MM', 'MIX-FO-MM'))


class Elimination(MutexFeature):
    @classmethod
    def create(cls, type):
        super().create(type)
        type = type.upper()
        if type not in ELIMINATION_TYPES:
            raise ValueError(f'Unknown `type`: got {type}')
        return cls(type=type)

    def get_complexity(self):
        order = {'FO': 0, 'ZO': 1, 'MM': 2, 'MIX-FO-MM': 3}
        return order[self.type]
