from .mutex_feature import MutexFeature

METABOLITE_TYPES = frozenset(('PSC', 'BASIC'))


class Metabolite(MutexFeature):
    @classmethod
    def create(cls, type):
        super().create(type)
        type = type.upper()
        if type not in METABOLITE_TYPES:
            raise ValueError(f'Unknown `type`: got {type}')
        return cls(type=type)

    def get_complexity(self):
        order = {'PSC': 0, 'BASIC': 1}
        return order[self.type]
