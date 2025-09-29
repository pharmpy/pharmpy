import itertools

from lark.visitors import Interpreter

from pharmpy.mfl.features.peripherals import PERIPHERAL_TYPES

from ..features import Absorption, LagTime, Peripherals, Transits
from ..features.absorption import ABSORPTION_TYPES


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
