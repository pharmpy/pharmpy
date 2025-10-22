import itertools
from typing import Union

from lark.visitors import Interpreter

from ..features import (
    ABSORPTION_TYPES,
    COVARIANCE_TYPES,
    COVARIATE_FP_TYPES,
    COVARIATE_OP_TYPES,
    DIRECT_EFFECT_TYPES,
    EFFECT_COMP_TYPES,
    ELIMINATION_TYPES,
    IIV,
    INDIRECT_EFFECT_TYPES,
    IOV,
    METABOLITE_TYPES,
    VARIABILITY_FP_TYPES,
    Absorption,
    Allometry,
    Covariance,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    Peripherals,
    Ref,
    Transits,
)


class MFLInterpreter(Interpreter):
    def __init__(self, definitions=None):
        self.definitions = definitions if definitions is not None else dict()
        super().__init__()

    def expand_ref(self, arg):
        values = self.definitions.get(arg.name)
        if values is not None:
            return values
        return [arg]

    def interpret(self, tree):
        return self.visit_children(tree)

    def definition(self, tree):
        return []

    def absorption(self, tree):
        return AbsorptionInterpreter().interpret(tree)

    def transits(self, tree):
        return TransitsInterpreter().interpret(tree)

    def peripherals(self, tree):
        return PeripheralsInterpreter().interpret(tree)

    def lagtime(self, tree):
        return LagTimeInterpreter().interpret(tree)

    def elimination(self, tree):
        return EliminationInterpreter().interpret(tree)

    def direct_effect(self, tree):
        return DirectEffectInterpreter().interpret(tree)

    def indirect_effect(self, tree):
        return IndirectEffectInterpreter().interpret(tree)

    def effect_comp(self, tree):
        return EffectCompInterpreter().interpret(tree)

    def metabolite(self, tree):
        return MetaboliteInterpreter().interpret(tree)

    def allometry(self, tree):
        return AllometryInterpreter().interpret(tree)

    def covariate(self, tree):
        return CovariateInterpreter(self.definitions).interpret(tree)

    def iiv(self, tree):
        return IIVInterpreter(self.definitions).interpret(tree)

    def iov(self, tree):
        return IOVInterpreter(self.definitions).interpret(tree)

    def covariance(self, tree):
        return CovarianceInterpreter(self.definitions).interpret(tree)

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()

    def wildcard(self, tree):
        return ['*']

    def optional(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return True

    def ref(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        name = children[0].value
        assert isinstance(name, str)
        return Ref(name)


class AbsorptionInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        validate_values(types, ABSORPTION_TYPES, 'ABSORPTION')
        absorptions = [Absorption.create(type=type) for type in types]
        return sorted(absorptions)

    def wildcard(self, tree) -> list[str]:
        return sorted(ABSORPTION_TYPES)


class CountInterpreter(MFLInterpreter):
    def count(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        child = children[0]
        return child if isinstance(child, list) else [child]

    def range(self, tree) -> list[int]:
        children = self.visit_children(tree)
        assert len(children) == 2
        left, right = children
        assert isinstance(left, int)
        assert isinstance(right, int)
        return list(range(left, right + 1))

    def number(self, tree) -> int:
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return int(value)


class PeripheralsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        numbers = children[0]
        if len(children) == 1:
            peripherals = [Peripherals.create(number=n) for n in numbers]
            return sorted(peripherals)

        types = children[1]
        validate_values(types, ['MET', 'DRUG'], 'PERIPHERALS')

        peripherals = []
        for number, type in itertools.product(numbers, types):
            metabolite = True if type.upper() == 'MET' else False
            p = Peripherals.create(number=number, metabolite=metabolite)
            peripherals.append(p)
        return sorted(peripherals)

    def wildcard(self, tree) -> list[str]:
        return ['MET', 'DRUG']


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        numbers = children[0]
        if len(children) == 1:
            transits = [Transits.create(number=n) for n in numbers]
            return sorted(transits)

        depot_settings = children[1]
        validate_values(depot_settings, ['DEPOT', 'NODEPOT'], 'TRANSITS')

        transits = []
        for number, depot in itertools.product(numbers, depot_settings):
            depot = True if depot.upper() == 'DEPOT' else False
            t = Transits.create(number=number, depot=depot)
            transits.append(t)
        return sorted(transits)

    def n(self, tree) -> str:
        return 'N'

    def wildcard(self, tree) -> list[str]:
        return ['DEPOT', 'NODEPOT']


class LagTimeInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1

        validate_values(children[0], ['ON', 'OFF'], 'LAGTIME')
        on_off = [True if val.upper() == 'ON' else False for val in children[0]]

        lagtimes = [LagTime.create(on=type) for type in on_off]
        return sorted(lagtimes)

    def wildcard(self, tree) -> list[str]:
        return ['ON', 'OFF']


class EliminationInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        validate_values(types, ELIMINATION_TYPES, 'ELIMINATION')
        eliminations = [Elimination.create(type=type) for type in types]
        return sorted(eliminations)

    def wildcard(self, tree) -> list[str]:
        return sorted(ELIMINATION_TYPES)


class DirectEffectInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        validate_values(types, DIRECT_EFFECT_TYPES, 'DIRECTEFFECT')
        direct_effects = [DirectEffect.create(type=type) for type in types]
        return sorted(direct_effects)

    def wildcard(self, tree) -> list[str]:
        return sorted(DIRECT_EFFECT_TYPES)


class IndirectEffectInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2

        types = self.expand_indirect(children[0])
        validate_values(types, INDIRECT_EFFECT_TYPES, 'INDIRECTEFFECT')

        production_types = self.expand_production(children[1])
        validate_values(production_types, ['PRODUCTION', 'DEGRADATION'], 'INDIRECTEFFECT')

        indirect_effects = []
        for type, production_type in itertools.product(types, production_types):
            production = True if production_type.upper() == 'PRODUCTION' else False
            indirect_effects.append(IndirectEffect.create(type=type, production=production))
        return sorted(indirect_effects)

    @staticmethod
    def expand_indirect(arg: list[str]) -> list[str]:
        if arg == ['*']:
            return sorted(INDIRECT_EFFECT_TYPES)
        else:
            return arg

    @staticmethod
    def expand_production(arg: list[str]) -> list[str]:
        if arg == ['*']:
            return ['PRODUCTION', 'DEGRADATION']
        else:
            return arg


class EffectCompInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        validate_values(types, EFFECT_COMP_TYPES, 'EFFECTCOMP')
        effects_comps = [EffectComp.create(type=type) for type in types]
        return sorted(effects_comps)

    def wildcard(self, tree) -> list[str]:
        return sorted(EFFECT_COMP_TYPES)


class MetaboliteInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        validate_values(types, METABOLITE_TYPES, 'METABOLITE')
        metabolites = [Metabolite.create(type=type) for type in types]
        return sorted(metabolites)

    def wildcard(self, tree) -> list[str]:
        return sorted(METABOLITE_TYPES)


class AllometryInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        kwargs = {'covariate': children[0]}
        if len(children) == 2:
            kwargs['reference'] = children[1]
        return [Allometry.create(**kwargs)]

    def value(self, tree) -> str:
        return tree.children[0].value

    def decimal(self, tree) -> Union[float, int]:
        return float(tree.children[0].value)


class CovariateInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 3 <= len(children) <= 5
        if not isinstance(children[0], bool):
            children.insert(0, False)
        if len(children) == 4:
            children.append('*')
        assert len(children) == 5

        is_optional = children[0]
        params = self.expand(children[1], wildcard=[Ref('pop_params')])
        covs = self.expand(children[2], wildcard=[Ref('covariates')])
        fps = self.expand(children[3], wildcard=sorted(COVARIATE_FP_TYPES))
        validate_values(fps, COVARIATE_FP_TYPES, 'COVARIATE')
        ops = children[4]
        validate_values(ops, COVARIATE_OP_TYPES, 'COVARIATE')

        effects = []
        for param, cov, fp, op in itertools.product(params, covs, fps, ops):
            effect = Covariate.create(
                parameter=param, covariate=cov, fp=fp, op=op, optional=is_optional
            )
            effects.append(effect)

        return sorted(effects)

    def op_option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value

    def expand(self, arg, wildcard):
        if arg == ['*']:
            return wildcard
        elif isinstance(arg, Ref):
            return self.expand_ref(arg)
        return arg


class VariabilityInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 2 <= len(children) <= 3
        if not isinstance(children[0], bool):
            children.insert(0, False)
        assert len(children) == 3

        is_optional = children[0]
        params = self.expand(children[1], wildcard=[Ref('pop_params')])
        fps = self.expand(children[2], wildcard=sorted(VARIABILITY_FP_TYPES))

        if isinstance(self, IIVInterpreter):
            type = 'IIV'
            func = IIV.create
        else:
            type = 'IOV'
            func = IOV.create

        validate_values(fps, VARIABILITY_FP_TYPES, type)

        effects = []
        for param, fp in itertools.product(params, fps):
            effect = func(parameter=param, fp=fp, optional=is_optional)
            effects.append(effect)

        return sorted(effects)

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()

    def expand(self, arg, wildcard):
        if arg == ['*']:
            return wildcard
        elif isinstance(arg, Ref):
            return self.expand_ref(arg)
        return arg


class IIVInterpreter(VariabilityInterpreter):
    pass


class IOVInterpreter(VariabilityInterpreter):
    pass


class CovarianceInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 2 <= len(children) <= 3
        if not isinstance(children[0], bool):
            children.insert(0, False)
        assert len(children) == 3

        is_optional = children[0]
        types = self.expand(children[1], wildcard=sorted(COVARIANCE_TYPES))
        validate_values(types, COVARIANCE_TYPES, 'COVARIANCE')
        params = self.expand(children[2], wildcard=[Ref('IIV')])
        param_pairs = self.get_pairwise_parameters(params)

        covariances = []
        for type, parameters in itertools.product(types, param_pairs):
            effect = Covariance.create(type=type, parameters=parameters, optional=is_optional)
            covariances.append(effect)

        return sorted(covariances)

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()

    def expand(self, arg, wildcard):
        if arg == ['*']:
            return wildcard
        elif isinstance(arg, Ref):
            return self.expand_ref(arg)
        return arg

    @staticmethod
    def get_pairwise_parameters(parameters):
        assert isinstance(parameters, list)
        if isinstance(parameters[0], Ref):
            return parameters
        else:
            return list(itertools.combinations(parameters, 2))


class DefinitionInterpreter(Interpreter):

    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        symbol = children[0].value
        values = children[1]
        return symbol, values

    def value(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return value.upper()


def validate_values(values, allowed_values, feature_name=''):
    allowed_values = list(allowed_values)
    not_valid = {val for val in values if val not in allowed_values}
    if not_valid:
        raise ValueError(
            f'Invalid values in {feature_name}: {sorted(not_valid)} (must be one of {allowed_values})'
        )
