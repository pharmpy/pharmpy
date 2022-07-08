from dataclasses import dataclass
from typing import List, Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature
from .symbols import Symbol, Wildcard


@dataclass
class CovariateEffects(ModelFeature):
    parameter: Union[Symbol, List[str]]
    covariate: Union[Symbol, List[str]]
    fp: List[str]
    op: Literal['*', '+'] = '*'


@dataclass
class Ref(Symbol):
    name: str


ParameterWildcard = Wildcard()
CovariateWildcard = Wildcard()
EffectFunctionWildcard = Wildcard()


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

    def fp_wildcard(self, tree):
        return EffectFunctionWildcard

    def ref(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        name = children[0].value
        assert isinstance(name, str)
        return Ref(name)
