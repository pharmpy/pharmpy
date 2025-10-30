from __future__ import annotations

import builtins
import itertools
from typing import TYPE_CHECKING, Mapping, Sequence, Union

from pharmpy.internals.set.subsets import subsets
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

    def expand(
        self, expand_to: Mapping[Ref, Sequence[Union[str, Sequence[str]]]]
    ) -> tuple[Covariance, ...]:
        if self.is_expanded():
            return (self,)

        assert isinstance(self.parameters, Ref)

        try:
            parameters = expand_to[self.parameters]
        except KeyError:
            raise ValueError(f'Ref not found in `expand_to`: {self.parameters}')

        if parameters:
            if isinstance(parameters[0], str):
                parameters = tuple(itertools.combinations(parameters, r=2))
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
            blocks = Covariance.get_covariance_blocks(features)
            for block in blocks:
                optional_str = '?' if optional else ''
                covariances.append(f'COVARIANCE{optional_str}({type},{get_repr(block)})')

        return ';'.join(covariances)

    @staticmethod
    def get_covariance_blocks(mf: Union[ModelFeatures, Sequence[Covariance]]):
        features = tuple(feat for feat in mf if isinstance(feat, Covariance))
        assert len(features) == len(mf)

        if any(feature.type != features[0].type for feature in features):
            raise ValueError('All features must have the same `type`')

        if len(features) == 1:
            return (features[0].parameters,)

        parameter_pairs = [
            feature.parameters for feature in features if not isinstance(feature.parameters, Ref)
        ]
        unique_parameters = sorted(set(x for sub in parameter_pairs for x in sub))

        # Blocks of size 2 is trivial since each Covariance object represents this
        possible_subsets = list(subsets(unique_parameters, min_size=3))
        # Sort by size, biggest first
        possible_subsets.sort(key=lambda x: len(x), reverse=True)

        found_subsets = []
        params_in_block = []
        while possible_subsets:
            if len(params_in_block) == len(unique_parameters):
                break

            current_subset = possible_subsets.pop(0)
            if set(current_subset).intersection(params_in_block):
                continue

            relevant_pairs = {
                tuple(sorted(p)) for p in parameter_pairs if set(p).intersection(current_subset)
            }
            relevant_params = {x for sub in relevant_pairs for x in sub}
            pairs_in_subset = {
                tuple(sorted(p)) for p in itertools.combinations(relevant_params, r=2)
            }

            if relevant_pairs == pairs_in_subset:
                for pair in relevant_pairs:
                    a, b = pair
                    parameter_pairs.remove((a, b))
                found_subsets.append(current_subset)
                params_in_block.extend(current_subset)

        parameter_pairs_ref = [
            feature.parameters for feature in features if isinstance(feature.parameters, Ref)
        ]
        if parameter_pairs_ref:
            blocks = tuple(parameter_pairs_ref) + tuple(found_subsets) + tuple(parameter_pairs)
        else:
            blocks = tuple(found_subsets) + tuple(parameter_pairs)

        return blocks
