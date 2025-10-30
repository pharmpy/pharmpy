from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence, Union

from pharmpy.mfl.features.help_functions import get_repr, group_args

from .model_feature import ModelFeature
from .symbols import Ref

if TYPE_CHECKING:
    from ..model_features import ModelFeatures

VARIABILITY_FP_TYPES = frozenset(('EXP', 'ADD', 'PROP', 'LOG', 'RE_LOG'))


class Variability(ModelFeature):
    def __init__(self, parameter: Union[str, Ref], fp: str, optional: bool):
        self._parameter = parameter
        self._fp = fp
        self._optional = optional

    @classmethod
    def create(
        cls, parameter: Union[str, Ref], fp: str = 'EXP', optional: bool = False
    ) -> Variability:
        if not isinstance(parameter, str) and not isinstance(parameter, Ref):
            raise TypeError(f'Type of `parameter` must be a string or Ref: got {type(parameter)}')
        if not isinstance(optional, bool):
            raise TypeError(f'Type of `optional` must be a bool: got {type(optional)}')

        parameter = parameter.upper() if isinstance(parameter, str) else parameter
        fp = cls._canonicalize_type(fp, VARIABILITY_FP_TYPES, 'fp')
        return cls(parameter, fp, optional)

    def replace(self, **kwargs):
        parameter = kwargs.get('parameter', self.parameter)
        fp = kwargs.get('fp', self.fp)
        optional = kwargs.get('optional', self.optional)
        return self.__class__.create(parameter=parameter, fp=fp, optional=optional)

    @property
    def parameter(self) -> Union[str, Ref]:
        return self._parameter

    @property
    def fp(self) -> str:
        return self._fp

    @property
    def optional(self) -> bool:
        return self._optional

    @property
    def args(self) -> tuple[Union[str, Ref], str, bool]:
        return self.parameter, self.fp, self.optional

    def expand(self, expand_to: Mapping[Ref, Sequence[str]]) -> tuple[Variability, ...]:
        if self.is_expanded():
            return (self,)

        try:
            assert isinstance(self.parameter, Ref)
            parameters = expand_to[self.parameter]
        except KeyError:
            raise ValueError(f'Ref not found in `expand_to`: {self.parameter}')

        if parameters:
            effects = [self.replace(parameter=p) for p in parameters]
            return tuple(sorted(effects))
        else:
            return tuple()

    def __repr__(self) -> str:
        optional = '?' if self.optional else ''
        class_name = self.__class__.__name__.upper()
        return f'{class_name}{optional}({self.parameter},{self.fp})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        return (
            self.parameter == other.parameter
            and self.fp == other.fp
            and self.optional == other.optional
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            return False
        if self.optional != other.optional:
            return self.optional < other.optional
        if self.parameter != other.parameter:
            return str(self.parameter) < str(other.parameter)
        return self.fp < other.fp

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, Variability))

        assert len(features) == len(mf.features)

        if len(features) == 1:
            return repr(features[0])

        no_of_args = len(features[0].args)
        args_grouped = group_args([feature.args for feature in features], i=no_of_args)

        class_name = features[0].__class__.__name__.upper()

        effects = []
        for arg in args_grouped:
            parameter, fp, optional = arg
            optional_repr = '?' if optional else ''
            inner = f'{get_repr(parameter)},{get_repr(fp)}'
            effects.append(f'{class_name}{optional_repr}({inner})')

        return ';'.join(effects)


class IIV(Variability):
    pass


class IOV(Variability):
    pass
