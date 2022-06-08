from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Literal, Tuple, Union

from lark import Lark
from lark.visitors import Interpreter

from pharmpy.model import Model

EffectLiteral = Tuple[str, str, str, str]
EffectSpecFeature = Union[str, Tuple[str, ...]]
Spec = Tuple[EffectSpecFeature, EffectSpecFeature, EffectSpecFeature, EffectSpecFeature]


def _ensure_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def parse_spec(spec: Iterable[Spec]) -> Iterable[EffectLiteral]:
    for parameters, covariates, fps, operations in spec:
        parameters = _ensure_tuple(parameters)
        covariates = _ensure_tuple(covariates)
        fps = _ensure_tuple(fps)
        operations = _ensure_tuple(operations)
        yield from product(parameters, covariates, fps, operations)


grammar = r"""
%ignore " "
start: statement (_SEPARATOR statement)*
_SEPARATOR: /;|\n/
statement:  alias | covariate

alias: absorption | elimination | distribution | categorical | continuous
absorption: "ABSORPTION"i "(" names ")"
elimination: "ELIMINATION"i "(" names ")"
distribution: "DISTRIBUTION"i "(" names ")"
categorical: "CATEGORICAL"i "(" names ")"
continuous: "CONTINUOUS"i "(" names ")"

covariate: "COVARIATE"i "(" parameter_option "," covariate_option "," fp_option ["," op_option] ")"
parameter_option: names
    | absorption_alias
    | elimination_alias
    | distribution_alias
    | parameter_wildcard
covariate_option: names
    | categorical_alias
    | continuous_alias
    | covariate_wildcard
fp_option: names
!op_option: "+" | "*"
absorption_alias: "@ABSORPTION"i
elimination_alias: "@ELIMINATION"i
distribution_alias: "@DISTRIBUTION"i
categorical_alias: "@CATEGORICAL"i
continuous_alias: "@CONTINUOUS"i
parameter_wildcard: WILDCARD
covariate_wildcard: WILDCARD

names: value | array
value: /[a-zA-Z0-9-]+/
array: "[" [value ("," value)*] "]"
WILDCARD: "*"
"""


class AliasInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return children[0]

    def absorption(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        names = children[0]
        return AbsorptionAlias(names)

    def elimination(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        names = children[0]
        return EliminationAlias(names)

    def distribution(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        names = children[0]
        return DistributionAlias(names)

    def categorical(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        names = children[0]
        return CategoricalAlias(names)

    def continuous(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        names = children[0]
        return ContinuousAlias(names)

    def names(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        child = children[0]
        return child if isinstance(child, list) else [child]

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()


class CovariateInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 3 <= len(children) <= 4
        return CovariateEffects(*children)

    def option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        child = children[0]
        return child if isinstance(child, (Symbol, list)) else [child]

    def parameter_option(self, tree):
        return self.option(tree)

    def covariate_option(self, tree):
        return self.option(tree)

    def fp_option(self, tree):
        return self.option(tree)

    def op_option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value

    def names(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        child = children[0]
        return child if isinstance(child, list) else [child]

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()

    def parameter_wildcard(self, tree):
        return ParameterWildcard

    def covariate_wildcard(self, tree):
        return CovariateWildcard

    def absorption_alias(self, tree):
        return AllAbsorptionParameters

    def elimination_alias(self, tree):
        return AllEliminationParameters

    def distribution_alias(self, tree):
        return AllDistributionParameters

    def categorical_alias(self, tree):
        return AllCategoricalCovariates

    def continuous_alias(self, tree):
        return AllContinuousCovariates


@dataclass
class Alias:
    names: List[str]


class AbsorptionAlias(Alias):
    pass


class EliminationAlias(Alias):
    pass


class DistributionAlias(Alias):
    pass


class CategoricalAlias(Alias):
    pass


class ContinuousAlias(Alias):
    pass


class Symbol:
    pass


class Wildcard(Symbol):
    pass


AllAbsorptionParameters = Symbol()
AllEliminationParameters = Symbol()
AllDistributionParameters = Symbol()
AllCategoricalCovariates = Symbol()
AllContinuousCovariates = Symbol()

ParameterWildcard = Wildcard()
CovariateWildcard = Wildcard()


@dataclass
class CovariateEffects:
    parameter: Union[Symbol, List[str]]
    covariate: Union[Symbol, List[str]]
    fp: List[str]
    op: Literal['*', '+'] = '*'


class DSLInterpreter(Interpreter):
    def interpret(self, tree):
        return self.visit_children(tree)

    def statement(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return children[0]

    def alias(self, tree):
        return AliasInterpreter().interpret(tree)

    def covariate(self, tree):
        return CovariateInterpreter().interpret(tree)


class Effects:
    def __init__(self, source: str):
        parser = Lark(
            grammar,
            parser='lalr',
            lexer='standard',
            propagate_positions=False,
            maybe_placeholders=False,
            cache=True,
        )
        tree = parser.parse(source)
        effects = []
        absorption = []
        elimination = []
        distribution = []
        categorical = []
        continuous = []
        for statement in DSLInterpreter().interpret(tree):
            if isinstance(statement, CovariateEffects):
                effects.append(statement)
            else:
                assert isinstance(statement, Alias)
                if isinstance(statement, AbsorptionAlias):
                    absorption.extend(statement.names)
                elif isinstance(statement, EliminationAlias):
                    elimination.extend(statement.names)
                elif isinstance(statement, DistributionAlias):
                    distribution.extend(statement.names)
                elif isinstance(statement, CategoricalAlias):
                    categorical.extend(statement.names)
                else:
                    assert isinstance(statement, ContinuousAlias)
                    continuous.extend(statement.names)

        self._effects = effects
        self._aliases = {
            'absorption': absorption,
            'elimination': elimination,
            'distribution': distribution,
            'categorical': categorical,
            'continuous': continuous,
        }

    def spec(self, model: Model) -> Iterable[Spec]:
        for effect in self._effects:
            yield self._effects_to_tuple(model, effect)

    def _effects_to_tuple(self, model: Model, effect: CovariateEffects) -> Spec:
        parameter = (
            self._interpret_symbol(model, effect.parameter)
            if isinstance(effect.parameter, Symbol)
            else effect.parameter
        )
        covariate = (
            self._interpret_symbol(model, effect.covariate)
            if isinstance(effect.covariate, Symbol)
            else effect.covariate
        )
        fp = effect.fp
        op = effect.op
        return (tuple(parameter), tuple(covariate), tuple(f.lower() for f in fp), op)

    def _interpret_symbol(self, model: Model, symbol: Symbol) -> List[str]:
        if symbol is AllAbsorptionParameters:
            return self._aliases['absorption']

        if symbol is AllEliminationParameters:
            return self._aliases['elimination']

        if symbol is AllDistributionParameters:
            return self._aliases['distribution']

        if symbol is AllCategoricalCovariates:
            return self._aliases['categorical'] or [
                column.name
                for column in model.datainfo
                if column.type == 'covariate' and not column.continuous
            ]

        if symbol is AllContinuousCovariates:
            return self._aliases['continuous'] or [
                column.name
                for column in model.datainfo
                if column.type == 'covariate' and column.continuous
            ]

        assert isinstance(symbol, Wildcard)

        if symbol is ParameterWildcard:
            # TODO Only select parameters with IIVs
            return [parameter.name for parameter in model.parameters]

        assert symbol is CovariateWildcard

        return [column.name for column in model.datainfo if column.type == 'covariate']
