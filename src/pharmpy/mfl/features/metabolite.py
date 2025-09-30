from .mutex_feature import MutexFeature

METABOLITE_TYPES = frozenset(('PSC', 'BASIC'))


class Metabolite(MutexFeature):
    @classmethod
    def create(cls, type):
        type = cls._canonicalize_type(type, METABOLITE_TYPES)
        return cls(type=type)

    def get_complexity(self):
        order = {'PSC': 0, 'BASIC': 1}
        return order[self.type]
