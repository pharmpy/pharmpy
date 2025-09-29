from collections import defaultdict

from .model_feature import ModelFeature

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
        if not isinstance(parameter, str):
            raise TypeError(f'Type of `parameter` must be a string: got {type(parameter)}')
        if not isinstance(covariate, str):
            raise TypeError(f'Type of `covariate` must be a string: got {type(covariate)}')
        if not isinstance(fp, str):
            raise TypeError(f'Type of `fp` must be a string: got {type(fp)}')
        if fp.upper() not in FP_TYPES:
            raise ValueError(f'Unknown `fp`: got {fp}')
        if not isinstance(op, str):
            raise TypeError(f'Type of `op` must be a string: got {type(op)}')
        if op not in OP_TYPES:
            raise ValueError(f'Unknown `op`: got {op}')
        if not isinstance(optional, bool):
            raise TypeError(f'Type of `optional` must be a bool: got {type(optional)}')

        return cls(
            parameter=parameter.upper(),
            covariate=covariate.upper(),
            fp=fp.upper(),
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
            return self.parameter < other.parameter
        if self.covariate != other.covariate:
            return self.covariate < other.covariate
        if self.fp != other.fp:
            return self.fp < other.fp
        return self.op < other.op

    @staticmethod
    def repr_many(features):
        features = sorted(features)
        if len(features) == 1:
            return repr(features[0])

        no_of_args = len(features[0].args)
        args_grouped = _group_args([feature.args for feature in features], i=no_of_args)

        effects = []
        for arg in args_grouped:
            parameter, covariate, fp, op, optional = arg
            optional_repr = '?' if optional else ''
            inner = f'{_get_repr(parameter)},{_get_repr(covariate)},{_get_repr(fp)},{_get_repr(op)}'
            effects.append(f'COVARIATE{optional_repr}({inner})')

        return ';'.join(effects)


def _group_args(args, i):
    if i == 0:
        return args

    groups = defaultdict(list)
    for a in args:
        head, tail = a[0:i], a[i:]
        groups[tail].append(head)

    args_new = []
    for tail, heads in groups.items():
        heads_grouped = defaultdict(list)
        for head in heads:
            heads_grouped[head[:-1]].append(head[-1])

        # Heads could not be grouped
        if len(heads_grouped) == len(heads):
            new = tuple(head + tail for head in heads)
            args_new.extend(new)
            continue
        for head, group in heads_grouped.items():
            head_new = []
            if head:
                head_new.append(head[0] if len(head) == 1 else tuple(head))
            head_new.append(group[0] if len(group) == 1 else tuple(group))
            args_new.append(tuple(head_new) + tail)

    return _group_args(tuple(args_new), i - 1)


def _get_repr(arg):
    if isinstance(arg, tuple):
        return f"[{','.join(arg)}]"
    else:
        return arg
