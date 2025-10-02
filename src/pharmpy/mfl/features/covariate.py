from .help_functions import get_repr, group_args
from .model_feature import ModelFeature
from .symbols import Ref

FP_TYPES = frozenset(('LIN', 'PIECE_LIN', 'EXP', 'POW', 'CAT', 'CAT2'))
OP_TYPES = frozenset(('+', '*'))


class Covariate(ModelFeature):
    def __init__(self, parameter, covariate, fp, op, optional):
        self._parameter = parameter
        self._covariate = covariate
        self._fp = fp
        self._op = op
        self._optional = optional

    @classmethod
    def create(cls, parameter, covariate, fp, op='*', optional=False):
        if not isinstance(parameter, str) and not isinstance(parameter, Ref):
            raise TypeError(f'Type of `parameter` must be a string or Ref: got {type(parameter)}')
        if not isinstance(covariate, str) and not isinstance(covariate, Ref):
            raise TypeError(f'Type of `covariate` must be a string or Ref: got {type(covariate)}')
        if not isinstance(optional, bool):
            raise TypeError(f'Type of `optional` must be a bool: got {type(optional)}')

        parameter = parameter.upper() if isinstance(parameter, str) else parameter
        covariate = covariate.upper() if isinstance(covariate, str) else covariate

        fp = cls._canonicalize_type(fp, FP_TYPES, 'fp')
        op = cls._canonicalize_type(op, OP_TYPES, 'op')

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
    def parameter(self):
        return self._parameter

    @property
    def covariate(self):
        return self._covariate

    @property
    def fp(self):
        return self._fp

    @property
    def op(self):
        return self._op

    @property
    def optional(self):
        return self._optional

    @property
    def args(self):
        return self.parameter, self.covariate, self.fp, self.op, self.optional

    def __repr__(self):
        optional = '?' if self.optional else ''
        inner = f'{self.parameter},{self.covariate},{self.fp},{self.op}'
        return f'COVARIATE{optional}({inner})'

    def __eq__(self, other):
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

    def __lt__(self, other):
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
    def repr_many(mfl):
        features = mfl.features
        if len(features) == 1:
            return repr(features[0])

        features = sorted(features)
        no_of_args = len(features[0].args)
        args_grouped = group_args([feature.args for feature in features], i=no_of_args)

        effects = []
        for arg in args_grouped:
            parameter, covariate, fp, op, optional = arg
            optional_repr = '?' if optional else ''
            inner = f'{get_repr(parameter)},{get_repr(covariate)},{get_repr(fp)},{get_repr(op)}'
            effects.append(f'COVARIATE{optional_repr}({inner})')

        return ';'.join(effects)
