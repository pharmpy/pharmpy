from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Union

from .help_functions import get_repr, group_args
from .model_feature import ModelFeature
from .symbols import Ref

if TYPE_CHECKING:
    from ..model_features import ModelFeatures


COVARIATE_FP_TYPES = frozenset(('LIN', 'PIECE_LIN', 'EXP', 'POW', 'CAT', 'CAT2'))
COVARIATE_OP_TYPES = frozenset(('+', '*'))


class Covariate(ModelFeature):
    def __init__(
        self,
        parameter: Union[str, Ref],
        covariate: Union[str, Ref],
        fp: str,
        op: str,
        optional: bool,
    ):
        self._parameter = parameter
        self._covariate = covariate
        self._fp = fp
        self._op = op
        self._optional = optional

    @classmethod
    def create(
        cls,
        parameter: Union[str, Ref],
        covariate: Union[str, Ref],
        fp: str,
        op: str = '*',
        optional: bool = False,
    ) -> Covariate:
        if not isinstance(parameter, str) and not isinstance(parameter, Ref):
            raise TypeError(f'Type of `parameter` must be a string or Ref: got {type(parameter)}')
        if not isinstance(covariate, str) and not isinstance(covariate, Ref):
            raise TypeError(f'Type of `covariate` must be a string or Ref: got {type(covariate)}')
        if not isinstance(optional, bool):
            raise TypeError(f'Type of `optional` must be a bool: got {type(optional)}')

        parameter = parameter.upper() if isinstance(parameter, str) else parameter
        covariate = covariate.upper() if isinstance(covariate, str) else covariate

        fp = cls._canonicalize_type(fp, COVARIATE_FP_TYPES, 'fp')
        op = cls._canonicalize_type(op, COVARIATE_OP_TYPES, 'op')

        return cls(
            parameter=parameter,
            covariate=covariate,
            fp=fp,
            op=op,
            optional=optional,
        )

    def replace(self, **kwargs):
        parameter = kwargs.get('parameter', self.parameter)
        covariate = kwargs.get('covariate', self.covariate)
        fp = kwargs.get('fp', self.fp)
        op = kwargs.get('op', self.op)
        optional = kwargs.get('optional', self.optional)

        return Covariate.create(
            parameter=parameter, covariate=covariate, fp=fp, op=op, optional=optional
        )

    @property
    def parameter(self) -> Union[str, Ref]:
        return self._parameter

    @property
    def covariate(self) -> Union[str, Ref]:
        return self._covariate

    @property
    def fp(self) -> str:
        return self._fp

    @property
    def op(self) -> str:
        return self._op

    @property
    def optional(self) -> bool:
        return self._optional

    @property
    def args(self) -> tuple[Union[str, Ref], Union[str, Ref], str, str, bool]:
        return self.parameter, self.covariate, self.fp, self.op, self.optional

    def expand(self, expand_to: Mapping[Ref, Sequence[str]]) -> tuple[Covariate, ...]:
        if self.is_expanded():
            return (self,)

        def _expanded(attr) -> Optional[tuple[str, ...]]:
            if isinstance(attr, Ref):
                values = expand_to.get(attr)
                if values is None:
                    raise ValueError(f'Ref not found in `expand_to`: {attr}')
                else:
                    return tuple(values)
            return None

        parameters = _expanded(self.parameter)
        covariates = _expanded(self.covariate)

        if (
            parameters is not None
            and len(parameters) == 0
            or covariates is not None
            and len(covariates) == 0
        ):
            return tuple()
        elif covariates and parameters:
            effects = [
                self.replace(parameter=p, covariate=c)
                for p, c in itertools.product(parameters, covariates)
            ]
        elif parameters:
            effects = [self.replace(parameter=p) for p in parameters]
        else:
            assert covariates
            effects = [self.replace(covariate=c) for c in covariates]

        return tuple(sorted(effects))

    def __repr__(self) -> str:
        optional = '?' if self.optional else ''
        inner = f'{self.parameter},{self.covariate},{self.fp},{self.op}'
        return f'COVARIATE{optional}({inner})'

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, Covariate):
            return False
        return (
            self.parameter == other.parameter
            and self.covariate == other.covariate
            and self.fp == other.fp
            and self.op == other.op
            and self.optional == other.optional
        )

    def __lt__(self, other) -> bool:
        if not isinstance(other, Covariate):
            return NotImplemented
        if self == other:
            return False
        if self.optional != other.optional:
            return self.optional < other.optional
        if self.parameter != other.parameter:
            return str(self.parameter) < str(other.parameter)
        if self.covariate != other.covariate:
            return str(self.covariate) < str(other.covariate)
        if self.fp != other.fp:
            return self.fp < other.fp
        return self.op < other.op

    @staticmethod
    def repr_many(mf: ModelFeatures) -> str:
        features = tuple(feat for feat in mf.features if isinstance(feat, Covariate))
        assert len(features) == len(mf.features)

        if len(features) == 1:
            return repr(features[0])

        no_of_args = len(features[0].args)
        args_grouped = group_args([feature.args for feature in features], i=no_of_args)

        effects = []
        for arg in args_grouped:
            parameter, covariate, fp, op, optional = arg
            optional_repr = '?' if optional else ''
            inner = f'{get_repr(parameter)},{get_repr(covariate)},{get_repr(fp)},{get_repr(op)}'
            effects.append(f'COVARIATE{optional_repr}({inner})')

        return ';'.join(effects)
