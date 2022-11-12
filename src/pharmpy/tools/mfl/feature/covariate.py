from itertools import product
from typing import Iterable, Tuple, TypeVar, Union

from pharmpy.model import Model
from pharmpy.modeling.covariate_effect import EffectType, OperationType, add_covariate_effect
from pharmpy.modeling.expressions import get_individual_parameters, get_pk_parameters

from ..statement.definition import Let
from ..statement.feature.covariate import (
    Covariate,
    CovariateWildcard,
    EffectFunctionWildcard,
    ParameterWildcard,
    Ref,
)
from ..statement.feature.symbols import Symbol, Wildcard
from ..statement.statement import Statement
from .feature import Feature

T = TypeVar('T')

EffectLiteral = Tuple[str, str, EffectType, OperationType]
EffectSpecFeature = Union[T, Tuple[T, ...]]
Spec = Tuple[
    EffectSpecFeature[str],
    EffectSpecFeature[str],
    EffectSpecFeature[EffectType],
    EffectSpecFeature[OperationType],
]

all_continuous_covariate_effects = (
    'lin',
    'piece_lin',
    'exp',
    'pow',
)

all_categorical_covariate_effects = ('cat',)

all_covariate_effects = all_continuous_covariate_effects + all_categorical_covariate_effects


def features(model: Model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for args in parse_spec(spec(model, statements)):
        yield ('COVARIATE', *args), lambda model: add_covariate_effect(model, *args)


def spec(model: Model, statements: Iterable[Statement]) -> Iterable[Spec]:
    effects = []
    definition = {}
    for statement in statements:
        if isinstance(statement, Covariate):
            effects.append(statement)
        elif isinstance(statement, Let):
            definition[statement.name] = statement.value

    for effect in effects:
        t = _effects_to_tuple(model, definition, effect)
        if all(t):  # NOTE We do not yield empty products
            yield t


def _ensure_tuple_or_list(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def parse_spec(spec: Iterable[Spec]) -> Iterable[EffectLiteral]:
    for parameters, covariates, fps, operations in spec:
        parameters = _ensure_tuple_or_list(parameters)
        covariates = _ensure_tuple_or_list(covariates)
        fps = _ensure_tuple_or_list(fps)
        operations = _ensure_tuple_or_list(operations)
        yield from product(parameters, covariates, fps, operations)


def _effects_to_tuple(model: Model, definition, effect: Covariate) -> Spec:
    parameters = (
        _interpret_symbol(model, definition, effect.parameter)
        if isinstance(effect.parameter, Symbol)
        else effect.parameter
    )
    covariates = (
        _interpret_symbol(model, definition, effect.covariate)
        if isinstance(effect.covariate, Symbol)
        else effect.covariate
    )
    fps = all_continuous_covariate_effects if effect.fp is EffectFunctionWildcard else effect.fp
    ops = effect.op
    return (parameters, covariates, tuple(fp.lower() for fp in fps), ops)


def _interpret_symbol(model: Model, definition, symbol: Symbol) -> Tuple[str, ...]:
    if isinstance(symbol, Ref):
        try:
            return definition[symbol.name]
        except KeyError:
            if symbol.name in ['ABSORPTION', 'ELIMINATION', 'DISTRIBUTION']:
                return tuple(get_pk_parameters(model, kind=symbol.name.lower()))
            elif symbol.name == 'CATEGORICAL':
                return tuple(
                    column.name
                    for column in model.datainfo
                    if column.type == 'covariate' and not column.continuous
                )
            elif symbol.name == 'CONTINUOUS':
                return tuple(
                    column.name
                    for column in model.datainfo
                    if column.type == 'covariate' and column.continuous
                )
            elif symbol.name == 'IIV':
                return tuple(get_individual_parameters(model, level='iiv'))
            else:
                return ()

    assert isinstance(symbol, Wildcard)

    if symbol is ParameterWildcard:
        return tuple(get_pk_parameters(model))

    assert symbol is CovariateWildcard

    return tuple(column.name for column in model.datainfo if column.type == 'covariate')
