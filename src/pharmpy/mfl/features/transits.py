from __future__ import annotations

import builtins
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pharmpy.mfl.features.help_functions import format_numbers

from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class Transits(ModelFeature):
    def __init__(self, number: int | str, depot: bool):
        self._number = number
        self._depot = depot

    @classmethod
    def create(cls, number: int | Literal['N'], depot: bool | None = None) -> Transits:
        if isinstance(number, int):
            if number < 0:
                raise ValueError(f'Number of transits must be positive: got {number}')
        elif isinstance(number, str):
            n = number.upper()
            if n == 'N':
                number = n
            else:
                raise ValueError(f'Value of `number` must be "N" if string: got {number}')
        else:
            raise TypeError(
                f'Type of `number` must be an integer or string "N": got {builtins.type(number)}'
            )
        if depot is not None and not isinstance(depot, bool):
            raise TypeError(f'Type of `depot` must be a bool: got {builtins.type(type)}')
        elif number == 'N' and depot:
            raise ValueError(f'Value of `depot` must be false if "N" transits: got {depot}')
        if depot is None:
            depot = bool(number != 'N')
        return cls(number=number, depot=depot)

    def replace(self, **kwargs):
        number = kwargs.get('number', self.number)
        depot = kwargs.get('depot', self.depot)
        return Transits.create(number=number, depot=depot)

    @property
    def number(self) -> int | str:
        return self._number

    @property
    def depot(self) -> bool:
        return self._depot

    @property
    def args(self) -> tuple[int | str, bool]:
        return self.number, self.depot

    def __repr__(self) -> str:
        inner = f'{self.number}'
        if isinstance(self.number, int) and not self.depot:
            inner += ',NODEPOT'
        return f'TRANSITS({inner})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Transits):
            return False
        return self.number == other.number and self.depot == other.depot

    def __lt__(self, other) -> bool:
        if not isinstance(other, Transits):
            return NotImplemented
        if self == other:
            return False

        if self.number != 'N' and other.number != 'N' and self.depot != other.depot:
            # Depot is "less then" no depot, False < True
            return self.depot > other.depot

        def _get_number(number) -> int:
            if number == 'N':
                return 9999
            else:
                return number

        return _get_number(self.number) < _get_number(other.number)

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, Transits))
        assert len(features) == len(mf.features)

        if len(features) == 1:
            return repr(features[0])
        features = sorted(features)
        transits_repr = []
        numbers_by_type = defaultdict(list)
        for feat in features:
            if feat.number == 'N':
                transits_repr.append(repr(feat))
                continue
            numbers_by_type[feat.depot] += [feat.number]
        numbers_by_type = {key: tuple(value) for key, value in numbers_by_type.items()}
        if len(numbers_by_type) > 1 and len(set(numbers_by_type.values())) == 1:
            numbers = next(iter(numbers_by_type.values()))
            inner = _get_inner(numbers, [True, False])
            return f'TRANSITS({inner})'

        for with_depot, numbers in numbers_by_type.items():
            numbers = list(numbers)
            inner = _get_inner(numbers, with_depot)
            transits_repr.append(f'TRANSITS({inner})')
        return ';'.join(transits_repr)


def _get_inner(numbers, with_depot):
    inner = format_numbers(numbers)
    if isinstance(with_depot, list):
        inner += ',[DEPOT,NODEPOT]'
    elif not with_depot:
        inner += ',NODEPOT'
    return inner
