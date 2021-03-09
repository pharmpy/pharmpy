# The modeling features language
# A high level language to describe model features and ranges of model features


# ; and \n separates features
# feature names are case insensitive
# FEATURE_NAME(options)     options is a comma separated list
# absorption(x)   - x can be FO, ZO, SEQ-ZO-FO, [FO, ZO] for multiple or * for all
# elimination(x)  - x can be FO, ZO, MM and MIX-FO-MM
# peripherals(n)  - n can be 0, 1, ... or a..b for a range (inclusive in both ends)
# transits(n)     - n as above


import functools

from lark import Lark
from lark.visitors import Interpreter

import pharmpy.modeling as modeling


grammar = r"""
start: feature (_SEPARATOR feature)*
feature: absorption | elimination | peripherals | transits

absorption: "ABSORPTION"i "(" (option) ")"
elimination: "ELIMINATION"i "(" (option) ")"
peripherals: "PERIPHERALS"i "(" (option) ")"
transits: "TRANSITS"i "(" (option) ")"
lagtime: "LAGTIME()"
option: (wildcard | range | value | array)
wildcard: "*"
range: NUMBER ".." NUMBER
value: /[a-zA-Z0-9-]+/
array: "[" [value ("," value)*] "]"
_SEPARATOR: /;|\n/
NUMBER: /\d+/
%ignore " "
"""


class ModelFeature:
    def apply_all(self, model):
        for name, func in self._funcs:
            transformed = model.copy()
            func(transformed)
            yield name, transformed


class Absorption(ModelFeature):
    def __init__(self, tree):
        self.args = OneArgInterpreter('absorption', ['FO', 'ZO', 'SEQ-ZO-FO']).interpret(tree)
        self._funcs = dict()
        for arg in self.args:
            name = f'ABSORPTION({arg})'
            if arg == 'FO':
                self._funcs[name] = modeling.first_order_absorption
            elif arg == 'ZO':
                self._funcs[name] = modeling.zero_order_absorption
            elif arg == 'SEQ-ZO-FO':
                self._funcs[name] = modeling.seq_zo_fo_absorption
            else:
                raise ValueError(f'Absorption {arg} not supported')


class Elimination(ModelFeature):
    def __init__(self, tree):
        self.args = OneArgInterpreter('elimination', ['FO', 'ZO', 'MM', 'MIX-FO-MM']).interpret(
            tree
        )
        self._funcs = dict()
        for arg in self.args:
            name = f'ELIMINATION({arg})'
            if arg == 'FO':
                self._funcs[name] = modeling.first_order_elimination
            elif arg == 'ZO':
                self._funcs[name] = modeling.zero_order_elimination
            elif arg == 'MM':
                self._funcs[name] = modeling.michaelis_menten_elimination
            elif arg == 'MIX-FO-MM':
                self._funcs[name] = modeling.mixed_mm_fo_elimination
            else:
                raise ValueError(f'Elimination {arg} not supported')


class Transits(ModelFeature):
    def __init__(self, tree):
        self.args = OneArgInterpreter('transits', []).interpret(tree)
        self._funcs = dict()
        for arg in self.args:
            name = f'TRANSITS({arg})'
            self._funcs[name] = functools.partial(modeling.set_transit_compartments, n=arg)


class Peripherals(ModelFeature):
    def __init__(self, tree):
        self.args = OneArgInterpreter('peripherals', []).interpret(tree)
        self._funcs = dict()
        for arg in self.args:
            name = f'PERIPHERALS({arg})'
            self._funcs[name] = None # FIXME: Need funcs to directly set number of peripherals


class Lagtime(ModelFeature):
    def __init__(self, tree):
        self._funcs = {'LAGTIME()': modeling.add_lag_time}


class OneArgInterpreter(Interpreter):
    def __init__(self, name, a):
        self.name = name
        self.all = a

    def visit_children(self, tree):
        a = super().visit_children(tree)
        return set().union(*a)

    def feature(self, tree):
        return self.visit_children(tree)

    def absorption(self, tree):
        if self.name == 'absorption':
            return self.visit_children(tree)
        else:
            return []

    def elimination(self, tree):
        if self.name == 'elimination':
            return self.visit_children(tree)
        else:
            return []

    def transits(self, tree):
        if self.name == 'transits':
            return self.visit_children(tree)
        else:
            return []

    def peripherals(self, tree):
        if self.name == 'peripherals':
            return self.visit_children(tree)
        else:
            return []

    def option(self, tree):
        return self.visit_children(tree)

    def value(self, tree):
        value = tree.children[0].value
        if self.name == 'transits':
            return {int(value)}
        else:
            return {value.upper()}

    def array(self, tree):
        return self.visit_children(tree)

    def wildcard(self, tree):
        if self.name == 'peripherals' or self.name == 'transits':
            raise ValueError(f'Wildcard (*) not supported for {self.name}')
        return set(self.all)

    def range(self, tree):
        return set(range(int(tree.children[0]), int(tree.children[1]) + 1))


class ModelFeatures:
    def __init__(self, code):
        parser = Lark(grammar)
        tree = parser.parse(code)
        if tree.find_data('absorption'):
            self.absorption = Absorption(tree)
        if tree.find_data('elimination'):
            self.elimination = Elimination(tree)
        if tree.find_data('transits'):
            self.transits = Transits(tree)
        if tree.find_data('peripherals'):
            self.peripherals = Peripherals(tree)
        if tree.find_data('lagtime'):
            self.lagtime = Lagtime(tree)
