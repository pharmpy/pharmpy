from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

from lark.visitors import Interpreter

from pharmpy.model import Model
from pharmpy.modeling import get_pk_parameters

from .feature import ModelFeature, feature
from .symbols import Symbol, Wildcard


@dataclass(frozen=True)
class Covariate(ModelFeature):
    parameter: Union[Symbol, Tuple[str, ...]]
    covariate: Union[Symbol, Tuple[str, ...]]
    fp: Tuple[str, ...]
    op: Literal['*', '+'] = '*'

    def eval(self, model: Optional[Model] = None):
        # Circular import issue
        from ...feature.covariate import _interpret_ref

        if model is not None:
            if self.parameter is ParameterWildcard:
                parameter = tuple(get_pk_parameters(model))
            elif isinstance(self.parameter, Ref):
                parameter = _interpret_ref(model, self.parameter)
            else:
                parameter = self.parameter

            if self.covariate is CovariateWildcard:
                covariate = tuple(
                    column.name for column in model.datainfo if column.type == 'covariate'
                )
            elif isinstance(self.covariate, Ref):
                covariate = _interpret_ref(model, self.covariate)
            else:
                covariate = self.covariate
        else:
            parameter = self.parameter
            covariate = self.covariate

        if self.fp == EffectFunctionWildcard:
            fp = (
                'lin',
                'piece_lin',
                'exp',
                'pow',
            )
        else:
            fp = self.fp

        op = self.op

        return Covariate(parameter=parameter, covariate=covariate, fp=fp, op=op)


@dataclass(frozen=True)
class Ref(Symbol):
    name: str


ParameterWildcard = Wildcard()
CovariateWildcard = Wildcard()
EffectFunctionWildcard = Wildcard()


class CovariateInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 3 <= len(children) <= 4
        return feature(Covariate, children)

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
