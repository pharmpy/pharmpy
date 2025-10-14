from __future__ import annotations

import builtins
from collections import defaultdict
from typing import TYPE_CHECKING

from .help_functions import format_numbers
from .model_feature import ModelFeature

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


class Peripherals(ModelFeature):
    def __init__(self, number: int, metabolite: bool):
        self._number = number
        self._metabolite = metabolite

    @classmethod
    def create(cls, number: int, metabolite: bool = False) -> Peripherals:
        if not isinstance(number, int):
            raise TypeError(f'Type of `number` must be an integer: got {builtins.type(number)}')
        if number < 0:
            raise ValueError(f'Number of peripherals must be positive: got {number}')
        if not isinstance(metabolite, bool):
            raise TypeError(f'Type of `metabolite` must be a bool: got {builtins.type(type)}')

        return cls(number=number, metabolite=metabolite)

    def replace(self, **kwargs):
        number = kwargs.get('number', self.number)
        metabolite = kwargs.get('metabolite', self.metabolite)
        return Peripherals.create(number=number, metabolite=metabolite)

    @property
    def number(self) -> int:
        return self._number

    @property
    def metabolite(self) -> bool:
        return self._metabolite

    @property
    def args(self) -> tuple[int, bool]:
        return self.number, self.metabolite

    def __repr__(self) -> str:
        inner = f'{self.number}'
        if self.metabolite:
            inner += ',MET'
        return f'PERIPHERALS({inner})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Peripherals):
            return False
        return self.number == other.number and self.metabolite == other.metabolite

    def __lt__(self, other) -> bool:
        if not isinstance(other, Peripherals):
            return NotImplemented
        if self == other:
            return False
        if self.metabolite != other.metabolite:
            return self.metabolite < other.metabolite
        return self.number < other.number

    @staticmethod
    def repr_many(mf: ModelFeatures):
        features = tuple(feat for feat in mf.features if isinstance(feat, Peripherals))
        assert len(features) == len(mf.features)

        if len(features) == 1:
            return repr(features[0])

        numbers_by_type = defaultdict(list)
        for feat in features:
            numbers_by_type[feat.metabolite].append(feat.number)

        peripherals_repr = []
        for metabolite, numbers in numbers_by_type.items():
            inner = format_numbers(numbers, as_range=True)
            if metabolite:
                inner += ',MET'
            peripherals_repr.append(f'PERIPHERALS({inner})')
        return ';'.join(peripherals_repr)
