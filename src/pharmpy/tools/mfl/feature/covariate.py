from itertools import product
from typing import Iterable, List, Tuple, TypeVar, Union

from pharmpy.model import Model
from pharmpy.modeling.covariate_effect import EffectType, OperationType, add_covariate_effect
from pharmpy.modeling.expressions import get_individual_parameters, get_pk_parameters

from ..statement.definition import Definition
from ..statement.feature.covariate import (
    CovariateEffects,
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
        if isinstance(statement, CovariateEffects):
            effects.append(statement)
        elif isinstance(statement, Definition):
            definition[statement.name] = statement.value

    for effect in effects:
        yield _effects_to_tuple(model, definition, effect)


def _ensure_tuple_or_list(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def parse_spec(spec: Iterable[Spec]) -> Iterable[EffectLiteral]:
    for parameters, covariates, fps, operations in spec:
        parameters = _ensure_tuple_or_list(parameters)
        covariates = _ensure_tuple_or_list(covariates)
        fps = _ensure_tuple_or_list(fps)
        operations = _ensure_tuple_or_list(operations)
        yield from product(parameters, covariates, fps, operations)


def _effects_to_tuple(model: Model, definition, effect: CovariateEffects) -> Spec:
    parameter = (
        _interpret_symbol(model, definition, effect.parameter)
        if isinstance(effect.parameter, Symbol)
        else effect.parameter
    )
    covariate = (
        _interpret_symbol(model, definition, effect.covariate)
        if isinstance(effect.covariate, Symbol)
        else effect.covariate
    )
    fp = all_continuous_covariate_effects if effect.fp is EffectFunctionWildcard else effect.fp
    op = effect.op
    return (tuple(parameter), tuple(covariate), tuple(f.lower() for f in fp), op)


def _interpret_symbol(model: Model, definition, symbol: Symbol) -> List[str]:
    if isinstance(symbol, Ref):
        value = definition.get(symbol.name, [])
        if symbol.name in ['ABSORPTION', 'ELIMINATION', 'DISTRIBUTION']:
            return value or get_pk_parameters(model, kind=symbol.name.lower())
        elif symbol.name == 'CATEGORICAL':
            return value or [
                column.name
                for column in model.datainfo
                if column.type == 'covariate' and not column.continuous
            ]
        elif symbol.name == 'CONTINUOUS':
            return value or [
                column.name
                for column in model.datainfo
                if column.type == 'covariate' and column.continuous
            ]
        elif symbol.name == 'IIV':
            return get_individual_parameters(model, level='iiv')
        else:
            return value

    assert isinstance(symbol, Wildcard)

    if symbol is ParameterWildcard:
        return get_pk_parameters(model)

    assert symbol is CovariateWildcard

    return [column.name for column in model.datainfo if column.type == 'covariate']
