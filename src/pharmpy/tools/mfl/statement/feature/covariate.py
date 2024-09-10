from dataclasses import dataclass
from itertools import product
from typing import Literal, Optional, Union

from lark.visitors import Interpreter

from pharmpy.model import Model
from pharmpy.modeling import get_pk_parameters

from .feature import ModelFeature, feature
from .symbols import Option, Symbol, Wildcard


@dataclass(frozen=True)
class Covariate(ModelFeature):
    parameter: Union[Symbol, tuple[str, ...]]
    covariate: Union[Symbol, tuple[str, ...]]
    fp: tuple[str, ...]
    op: Literal['*', '+'] = '*'
    optional: Option = Option(False)

    def eval(self, model: Optional[Model] = None, explicit_covariates: Optional[set] = None):
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
                'LIN',
                'PIECE_LIN',
                'EXP',
                'POW',
            )
        else:
            fp = self.fp

        op = self.op

        optional = self.optional

        if explicit_covariates and isinstance(self.parameter, Ref):
            param_cov = set(product(parameter, covariate))
            explicit_to_remove = [
                p for p, c in param_cov.intersection(explicit_covariates) if c in covariate
            ]
            parameter = tuple([p for p in parameter if p not in explicit_to_remove])
        if len(parameter) == 0 or len(covariate) == 0:
            return None
        return Covariate(parameter=parameter, covariate=covariate, fp=fp, op=op, optional=optional)

    def get_length(self, model: Optional[Model] = None):
        self_eval = self.eval(model)
        if self_eval is None:
            return 0
        len_add = len(self_eval.covariate) * len(self_eval.parameter) * len(self_eval.fp)
        if self.optional.option:
            len_add *= 2
        return len_add


@dataclass(frozen=True)
class Ref(Symbol):
    name: str


ParameterWildcard = Wildcard()
CovariateWildcard = Wildcard()
EffectFunctionWildcard = Wildcard()


class CovariateInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 3 <= len(children) <= 5
        if isinstance(children[0], Option):
            if len(children) == 4:
                children.append('*')
            children.append(children.pop(0))
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
        children = self.visit_children(tree)
        return list((child.value.upper()) for child in children)

    def optional_cov(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return Option(True)

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
