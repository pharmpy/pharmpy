import builtins

from .model_feature import ModelFeature


class Transits(ModelFeature):
    def __init__(self, number, with_depot):
        self._number = number
        self._with_depot = with_depot

    @classmethod
    def create(cls, number, with_depot=True):
        if not isinstance(number, int):
            raise TypeError(f'Type of `number` must be an integer: got {builtins.type(number)}')
        if number < 0:
            raise ValueError(f'Number of peripherals must be positive: got {number}')
        if not isinstance(with_depot, bool):
            raise TypeError(f'Type of `with_depot` must be a bool: got {builtins.type(type)}')
        return cls(number=number, with_depot=with_depot)

    def replace(self, **kwargs):
        number = kwargs.get('number', self.number)
        with_depot = kwargs.get('with_depot', self.with_depot)
        return Transits.create(number=number, with_depot=with_depot)

    @property
    def number(self):
        return self._number

    @property
    def with_depot(self):
        return self._with_depot

    @property
    def args(self):
        return self.number, self.with_depot

    def expand(self, model):
        return self

    def __repr__(self):
        inner = f'{self.number}'
        if self.with_depot:
            inner += ',DEPOT'
        else:
            inner += ',NODEPOT'
        return f'TRANSITS({inner})'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Transits):
            return False
        return self.number == other.number and self.with_depot == other.with_depot

    def __lt__(self, other):
        if not isinstance(other, Transits):
            return NotImplemented
        if self == other:
            return False
        if self.with_depot != other.with_depot:
            # Depot is "less then" no depot, False < True
            return self.with_depot > other.with_depot
        return self.number < other.number

    @staticmethod
    def repr_many(features):
        if len(features) == 1:
            return repr(features[0])
        numbers_by_type = dict()
        features = sorted(features)
        for feat in features:
            if feat.with_depot not in numbers_by_type:
                numbers_by_type[feat.with_depot] = [feat.number]
            else:
                numbers_by_type[feat.with_depot].append(feat.number)
        transits_repr = []
        for with_depot, numbers in numbers_by_type.items():
            numbers_sorted = sorted(numbers)
            if len(numbers_sorted) == 1:
                inner = f'{numbers_sorted[0]}'
            else:
                inner = f"[{','.join(str(n) for n in numbers_sorted)}]"
            if with_depot:
                inner += ',DEPOT'
            else:
                inner += ',NODEPOT'
            transits_repr.append(f'TRANSITS({inner})')
        return ';'.join(transits_repr)
