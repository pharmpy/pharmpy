import builtins

from .model_feature import ModelFeature


class Transits(ModelFeature):
    def __init__(self, number, with_depot):
        self._number = number
        self._with_depot = with_depot

    @classmethod
    def create(cls, number, with_depot=True):
        if not isinstance(number, int):
            if isinstance(number, str):
                number = number.upper()
                if number != 'N':
                    raise ValueError(f'Value of `number` must be "N" if string: got {number}')
            else:
                raise TypeError(
                    f'Type of `number` must be an integer or string "N": got {builtins.type(number)}'
                )
        elif number < 0:
            raise ValueError(f'Number of transits must be positive: got {number}')
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

    def __repr__(self):
        inner = f'{self.number}'
        if not self.with_depot:
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

        def _get_number(number):
            if number == 'N':
                return 9999
            else:
                return number

        return _get_number(self.number) < _get_number(other.number)

    @staticmethod
    def repr_many(mfl):
        features = sorted(mfl.features)
        if len(features) == 1:
            return repr(features[0])
        numbers_by_type = dict()
        for feat in features:
            if feat.with_depot not in numbers_by_type:
                numbers_by_type[feat.with_depot] = [feat.number]
            else:
                numbers_by_type[feat.with_depot].append(feat.number)
        transits_repr = []
        for with_depot, numbers in numbers_by_type.items():
            if 'N' in numbers:
                inner = 'N'
                numbers.remove('N')
                if not with_depot:
                    inner += ',NODEPOT'
                transits_repr.append(f'TRANSITS({inner})')
            if not numbers:
                continue
            numbers_sorted = sorted(numbers)
            if len(numbers_sorted) == 1:
                inner = f'{numbers_sorted[0]}'
            else:
                inner = f"[{','.join(str(n) for n in numbers_sorted)}]"
            if not with_depot:
                inner += ',NODEPOT'
            transits_repr.append(f'TRANSITS({inner})')
        return ';'.join(transits_repr)
