import itertools
from typing import Iterable, TypeVar, overload

from lark.visitors import Interpreter

from ..features import (
    Absorption,
    Allometry,
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
from ..features.absorption import ABSORPTION_TYPES
from ..features.covariate import FP_TYPES
from ..features.direct_effect import DIRECT_EFFECT_TYPES
from ..features.effect_compartment import EFFECT_COMP_TYPES
from ..features.elimination import ELIMINATION_TYPES
from ..features.indirect_effect import INDIRECT_EFFECT_TYPES, PRODUCTION_TYPES
from ..features.metabolite import METABOLITE_TYPES
from ..features.peripherals import PERIPHERAL_TYPES

T = TypeVar('T')


class MFLInterpreter(Interpreter):
    def __init__(self, definitions=None):
        self.definitions = definitions
        super().__init__()

    @overload
    def expand(self, arg: list[T], expand_to: None) -> list[T]: ...

    @overload
    def expand(self, arg: list[T], expand_to: Ref) -> list[Ref]: ...

    @overload
    def expand(self, arg: list[T], expand_to: Iterable[T]) -> list[T]: ...

    def expand(self, arg, expand_to=None):
        if isinstance(arg[0], Ref) and self.definitions is not None:
            values = self.definitions.get(arg[0].name)
            if values:
                return values
        if arg[0] == '*' and expand_to is not None:
            if isinstance(expand_to, Ref):
                return [expand_to]
            return list(expand_to)
        return arg

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

    def wildcard(self, tree):
        return '*'


class AbsorptionInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = self.expand(children[0], ABSORPTION_TYPES)
        absorptions = [Absorption.create(type=type) for type in types]
        return sorted(absorptions)

    def absorption_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


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
        types = self.expand(children[1], PERIPHERAL_TYPES)
        peripherals = []
        for number, type in itertools.product(numbers, types):
            p = Peripherals.create(number=number, type=type)
            peripherals.append(p)
        return sorted(peripherals)

    def peripheral_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        numbers = children[0]
        if len(children) == 1:
            transits = [Transits.create(number=n) for n in numbers]
            return sorted(transits)
        depot_settings = self.expand(children[1], [True, False])
        transits = []
        for number, with_depot in itertools.product(numbers, depot_settings):
            t = Transits.create(number=number, with_depot=with_depot)
            transits.append(t)
        return sorted(transits)

    def depot_modes(self, tree):
        children = self.visit_children(tree)
        return list(True if child.value.upper() == 'DEPOT' else False for child in children)

    def n(self, tree):
        return 'N'


class LagTimeInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        on_off = self.expand(children[0], [True, False])
        lagtimes = [LagTime.create(on=type) for type in on_off]
        return sorted(lagtimes)

    def lagtime_modes(self, tree):
        children = self.visit_children(tree)
        return list(True if child.value.upper() == 'ON' else False for child in children)


class EliminationInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = self.expand(children[0], ELIMINATION_TYPES)
        eliminations = [Elimination.create(type=type) for type in types]
        return sorted(eliminations)

    def elimination_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class DirectEffectInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = self.expand(children[0], DIRECT_EFFECT_TYPES)
        direct_effects = [DirectEffect.create(type=type) for type in types]
        return sorted(direct_effects)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class IndirectEffectInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        types = self.expand(children[0], INDIRECT_EFFECT_TYPES)
        production_types = self.expand(children[1], PRODUCTION_TYPES)
        indirect_effects = []
        for type, production_type in itertools.product(types, production_types):
            indirect_effects.append(
                IndirectEffect.create(type=type, production_type=production_type)
            )
        return sorted(indirect_effects)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def production_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class EffectCompInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = self.expand(children[0], EFFECT_COMP_TYPES)
        effects_comps = [EffectComp.create(type=type) for type in types]
        return sorted(effects_comps)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class MetaboliteInterpreter(MFLInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = self.expand(children[0], METABOLITE_TYPES)
        metabolites = [Metabolite.create(type=type) for type in types]
        return sorted(metabolites)

    def metabolite_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)


class AllometryInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        kwargs = {'covariate': children[0]}
        if len(children) == 2:
            kwargs['reference'] = children[1]
        return [Allometry.create(**kwargs)]

    def value(self, tree):
        return tree.children[0].value

    def decimal(self, tree):
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
        params = self.expand(children[1], expand_to=Ref('pop_params'))
        covs = self.expand(children[2], expand_to=Ref('covariates'))
        fps = self.expand(children[3], FP_TYPES)
        ops = children[4]

        effects = []
        for param, cov, fp, op in itertools.product(params, covs, fps, ops):
            effect = Covariate.create(
                parameter=param, covariate=cov, fp=fp, op=op, optional=is_optional
            )
            effects.append(effect)

        return sorted(effects)

    def option(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        child = children[0]
        return child if isinstance(child, list) else [child]

    def parameter_option(self, tree):
        return self.option(tree)

    def covariate_option(self, tree):
        return self.option(tree)

    def fp_option(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def optional_cov(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        value = children[0].value
        assert isinstance(value, str)
        return True

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

    def ref(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        name = children[0].value
        assert isinstance(name, str)
        return Ref(name)


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
