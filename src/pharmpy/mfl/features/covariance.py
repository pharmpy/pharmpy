from __future__ import annotations

import builtins
import itertools
from typing import TYPE_CHECKING, Sequence, Union

from pharmpy.mfl.features.help_functions import get_repr

from .model_feature import ModelFeature, Ref

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


COVARIANCE_TYPES = frozenset(('IIV', 'IOV'))


class Covariance(ModelFeature):
    def __init__(self, type: str, parameters: Union[tuple[str, str], Ref], optional: bool):
        self._type = type
        self._parameters = parameters
        self._optional = optional

    @classmethod
    def create(
        cls, type: str, parameters: Union[Sequence[str], Ref], optional: bool = False
    ) -> Covariance:
        type = cls._canonicalize_type(type, COVARIANCE_TYPES)
        if not isinstance(parameters, Sequence) and not isinstance(parameters, Ref):
            raise TypeError(
                f'Type of `parameters` must be a list or Ref: got {builtins.type(parameters)}'
            )
        if isinstance(parameters, Sequence):
            if len(parameters) != 2:
                raise ValueError(f'Number of `parameters` must be 2: got {len(parameters)}')
            if not all(types := isinstance(p, str) for p in parameters):
                raise TypeError(
                    f'Type of all parameters in `parameters` must be a str: got {types}'
                )
            parameters = tuple(sorted(p.upper() for p in parameters))
            assert len(parameters) == 2
        if not isinstance(optional, bool):
            raise TypeError(f'Type of `optional` must be a bool: got {builtins.type(optional)}')

        return cls(type=type, parameters=parameters, optional=optional)

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        parameters = kwargs.get('parameters', self.parameters)
        optional = kwargs.get('optional', self.optional)
        return Covariance.create(type=type, parameters=parameters, optional=optional)

    @property
    def type(self) -> str:
        return self._type

    @property
    def parameters(self) -> Union[tuple[str, str], Ref]:
        return self._parameters

    @property
    def optional(self) -> bool:
        return self._optional

    @property
    def args(self) -> tuple[str, Union[tuple[str, str], Ref], bool]:
        return self.type, self.parameters, self.optional

    def expand(self, expand_to: dict[Ref, Sequence[str]]) -> tuple[Covariance, ...]:
        if self.is_expanded():
            return (self,)

        assert isinstance(self.parameters, Ref)

        try:
            parameters = expand_to[self.parameters]
        except KeyError:
            raise ValueError(f'Ref not found in `expand_to`: {self.parameters}')

        if parameters:
            effects = [self.replace(parameters=pair) for pair in parameters]
            return tuple(sorted(effects))
        else:
            return tuple()

    def __repr__(self) -> str:
        optional = '?' if self.optional else ''
        return f'COVARIANCE{optional}({self.type},{get_repr(self.parameters)})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Covariance):
            return False
        return (
            self.type == other.type
            and self.parameters == other.parameters
            and self.optional == other.optional
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, Covariance):
            return NotImplemented
        if self == other:
            return False
        if self.optional != other.optional:
            return self.optional < other.optional
        if self.type != other.type:
            return self.type < other.type
        if type(self.parameters) is not type(other.parameters):
            order = {Ref: 0, tuple: 1}
            return order[type(self.parameters)] < order[type(other.parameters)]
        if isinstance(self.parameters, Ref):
            return str(self.parameters) < str(other.parameters)
        assert not isinstance(other.parameters, Ref)
        return self.parameters < other.parameters

    @staticmethod
    def repr_many(mf: ModelFeatures):
        features = tuple(feat for feat in mf.features if isinstance(feat, Covariance))
        assert len(features) == len(mf.features)

        if len(features) == 1:
            return repr(features[0])

        types = sorted(COVARIANCE_TYPES)
        optional = (True, False)
        groups = {
            (t, o): [f for f in features if f.type == t and f.optional == o]
            for t in types
            for o in optional
        }
        covariances = []
        for (type, optional), features in groups.items():
            if not features:
                continue
            parameters = [feature.parameters for feature in features]
            # FIXME: make more general
            cartesian_product = get_cartesian_product(parameters)
            if cartesian_product is not None:
                optional_str = '?' if optional else ''
                covariances.append(
                    f'COVARIANCE{optional_str}({type},{get_repr(cartesian_product)})'
                )
            else:
                covariances.append(';'.join(repr(feature) for feature in features))

        return ';'.join(covariances)


def get_cartesian_product(L):
    values = sorted(set(x for sub in L for x in sub))
    combinations = list(itertools.combinations(values, 2))
    if combinations == L:
        return values
    else:
        return None
