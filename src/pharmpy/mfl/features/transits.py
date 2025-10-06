from __future__ import annotations

import builtins
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, Union

from pharmpy.mfl.features.help_functions import format_numbers

from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class Transits(ModelFeature):
    def __init__(self, number: Union[int, str], with_depot: bool):
        self._number = number
        self._with_depot = with_depot

    @classmethod
    def create(cls, number: Union[int, Literal['N']], with_depot: bool = True) -> Transits:
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
        if not isinstance(with_depot, bool):
            raise TypeError(f'Type of `with_depot` must be a bool: got {builtins.type(type)}')
        return cls(number=number, with_depot=with_depot)

    def replace(self, **kwargs):
        number = kwargs.get('number', self.number)
        with_depot = kwargs.get('with_depot', self.with_depot)
        return Transits.create(number=number, with_depot=with_depot)

    @property
    def number(self) -> Union[int, str]:
        return self._number

    @property
    def with_depot(self) -> bool:
        return self._with_depot

    @property
    def args(self) -> tuple[Union[int, str], bool]:
        return self.number, self.with_depot

    def __repr__(self) -> str:
        inner = f'{self.number}'
        if not self.with_depot:
            inner += ',NODEPOT'
        return f'TRANSITS({inner})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Transits):
            return False
        return self.number == other.number and self.with_depot == other.with_depot

    def __lt__(self, other) -> bool:
        if not isinstance(other, Transits):
            return NotImplemented
        if self == other:
            return False
        if self.with_depot != other.with_depot:
            # Depot is "less then" no depot, False < True
            return self.with_depot > other.with_depot

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
        numbers_by_type = defaultdict(tuple)
        for feat in features:
            numbers_by_type[feat.with_depot] += (feat.number,)
        if len(numbers_by_type) > 1 and len(set(numbers_by_type.values())) == 1:
            numbers = list(numbers_by_type.values())[0]
            inner = _get_inner(numbers, [True, False])
            return f'TRANSITS({inner})'

        transits_repr = []
        for with_depot, numbers in numbers_by_type.items():
            numbers = list(numbers)
            if 'N' in numbers:
                numbers.remove('N')
                inner = _get_inner('N', with_depot)
                transits_repr.append(f'TRANSITS({inner})')
            if not numbers:
                continue
            inner = _get_inner(numbers, with_depot)
            transits_repr.append(f'TRANSITS({inner})')
        return ';'.join(transits_repr)


def _get_inner(numbers, with_depot):
    if isinstance(numbers, str):
        inner = f'{numbers}'
    else:
        inner = format_numbers(numbers)
    if isinstance(with_depot, list):
        inner += ',[DEPOT,NODEPOT]'
    elif not with_depot:
        inner += ',NODEPOT'
    return inner
