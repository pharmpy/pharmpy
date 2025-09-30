import builtins

from .model_feature import ModelFeature

PERIPHERAL_TYPES = frozenset(('DRUG', 'MET'))


class Peripherals(ModelFeature):
    def __init__(self, number, type):
        self._number = number
        self._type = type

    @classmethod
    def create(cls, number, type='DRUG'):
        if not isinstance(number, int):
            raise TypeError(f'Type of `number` must be an integer: got {builtins.type(number)}')
        if number < 0:
            raise ValueError(f'Number of peripherals must be positive: got {number}')
        type = cls._canonicalize_type(type, PERIPHERAL_TYPES)
        return cls(number=number, type=type)

    def replace(self, **kwargs):
        number = kwargs.get('number', self.number)
        type = kwargs.get('type', self.type)
        return Peripherals.create(number=number, type=type)

    @property
    def number(self):
        return self._number

    @property
    def type(self):
        return self._type

    @property
    def args(self):
        return self.number, self.type

    def __repr__(self):
        inner = f'{self.number}'
        if self.type != 'DRUG':
            inner += f',{self.type}'
        return f'PERIPHERALS({inner})'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Peripherals):
            return False
        return self.number == other.number and self.type == other.type

    def __lt__(self, other):
        if not isinstance(other, Peripherals):
            return NotImplemented
        if self == other:
            return False
        if self.type != other.type:
            type_rank = {'DRUG': 0, 'MET': 1}
            return type_rank[self.type] < type_rank[other.type]
        return self.number < other.number

    @staticmethod
    def repr_many(mfl):
        features = sorted(mfl.features)
        if len(features) == 1:
            return repr(features[0])
        numbers_by_type = dict()
        for feat in features:
            if feat.type not in numbers_by_type:
                numbers_by_type[feat.type] = [feat.number]
            else:
                numbers_by_type[feat.type].append(feat.number)
        peripherals_repr = []
        for type, numbers in numbers_by_type.items():
            numbers_sorted = sorted(numbers)
            if len(numbers_sorted) == 1:
                inner = f'{numbers_sorted[0]}'
            elif all(b - a == 1 for a, b in zip(numbers_sorted, numbers_sorted[1:])):
                inner = f'{numbers[0]}..{numbers[-1]}'
            else:
                inner = f"[{','.join(str(n) for n in numbers_sorted)}]"
            if type != 'DRUG':
                inner += f',{type}'
            peripherals_repr.append(f'PERIPHERALS({inner})')
        return ';'.join(peripherals_repr)
