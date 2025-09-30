import itertools

from lark.visitors import Interpreter

from ..features import (
    Absorption,
    Allometry,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    Peripherals,
    Transits,
)
from ..features.absorption import ABSORPTION_TYPES
from ..features.direct_effect import DIRECT_EFFECT_TYPES
from ..features.effect_compartment import EFFECT_COMP_TYPES
from ..features.elimination import ELIMINATION_TYPES
from ..features.indirect_effect import INDIRECT_EFFECT_TYPES, PRODUCTION_TYPES
from ..features.metabolite import METABOLITE_TYPES
from ..features.peripherals import PERIPHERAL_TYPES


class MFLInterpreter(Interpreter):
    def interpret(self, tree):
        return self.visit_children(tree)

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


class AbsorptionInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        absorptions = [Absorption.create(type=type) for type in types]
        return sorted(absorptions)

    def absorption_types(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def absorption_wildcard(self, tree):
        return list(ABSORPTION_TYPES)


class CountInterpreter(Interpreter):
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
        peripherals = []
        for number, type in itertools.product(numbers, types):
            p = Peripherals.create(number=number, type=type)
            peripherals.append(p)
        return sorted(peripherals)

    def peripheral_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def peripheral_wildcard(self, tree):
        return list(PERIPHERAL_TYPES)


class TransitsInterpreter(CountInterpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert 1 <= len(children) <= 2
        numbers = children[0]
        if len(children) == 1:
            transits = [Transits.create(number=n) for n in numbers]
            return sorted(transits)

        depot_settings = children[1]
        transits = []
        for number, with_depot in itertools.product(numbers, depot_settings):
            t = Transits.create(number=number, with_depot=with_depot)
            transits.append(t)
        return sorted(transits)

    def depot_modes(self, tree):
        children = self.visit_children(tree)
        return list(True if child.value.upper() == 'DEPOT' else False for child in children)

    def depot_wildcard(self, tree):
        return [True, False]

    def n(self, tree):
        return 'N'


class LagTimeInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        on_off = children[0]
        lagtimes = [LagTime.create(on=type) for type in on_off]
        return sorted(lagtimes)

    def lagtime_modes(self, tree):
        children = self.visit_children(tree)
        return list(True if child.value.upper() == 'ON' else False for child in children)

    def lagtime_wildcard(self, tree):
        return [True, False]


class EliminationInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        eliminations = [Elimination.create(type=type) for type in types]
        return sorted(eliminations)

    def elimination_types(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def elimination_wildcard(self, tree):
        return list(ELIMINATION_TYPES)


class DirectEffectInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        direct_effects = [DirectEffect.create(type=type) for type in types]
        return sorted(direct_effects)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def pdtype_wildcard(self, tree):
        return list(DIRECT_EFFECT_TYPES)


class IndirectEffectInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 2
        types = children[0]
        production_types = children[1]
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

    def pdtype_wildcard(self, tree):
        return list(INDIRECT_EFFECT_TYPES)

    def production_wildcard(self, tree):
        return list(PRODUCTION_TYPES)


class EffectCompInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        effects_comps = [EffectComp.create(type=type) for type in types]
        return sorted(effects_comps)

    def pdtype_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def pdtype_wildcard(self, tree):
        return list(EFFECT_COMP_TYPES)


class MetaboliteInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        types = children[0]
        metabolites = [Metabolite.create(type=type) for type in types]
        return sorted(metabolites)

    def metabolite_modes(self, tree):
        children = self.visit_children(tree)
        return list(child.value.upper() for child in children)

    def metabolite_wildcard(self, tree):
        return list(METABOLITE_TYPES)


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
