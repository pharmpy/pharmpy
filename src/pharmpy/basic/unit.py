from __future__ import annotations

import re
from typing import Union

from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.unicode import int_to_superscript


class Unit:
    def __init__(self, source: Union[Unit, str]):
        if isinstance(source, Unit):
            self._expr = source._expr
        else:
            self._expr = sympy.sympify(source).subs(_unit_subs())

    def unicode(self) -> str:
        printer = UnitPrinter()
        return printer._print(self._expr)

    def serialize(self) -> str:
        return sympy.srepr(self._expr)

    @classmethod
    def deserialize(cls, s) -> Unit:
        return cls(s)

    @classmethod
    def unitless(cls) -> Unit:
        return cls("1")

    def __eq__(self, other):
        return isinstance(other, Unit) and self._expr == other._expr or self._expr == other

    def __hash__(self):
        return hash(self._expr)

    def __repr__(self):
        return repr(self._expr)

    def _sympify_(self):
        return self._expr


# Type hint for public functions taking an expression as input
TUnit = str | Unit

_unit_subs_cache = None


def _unit_subs():
    global _unit_subs_cache
    if _unit_subs_cache is None:
        subs = {}
        import sympy.physics.units as units

        for k, v in units.__dict__.items():
            if isinstance(v, sympy.Expr) and v.has(units.Unit):
                subs[sympy.Symbol(k)] = v

        _unit_subs_cache = subs

    return _unit_subs_cache


class UnitPrinter(sympy_printing.str.StrPrinter):
    """Print physical unit as unicode"""

    def _print_Mul(self, expr):
        pow_strings = [self._print(e) for e in expr.args if e.is_Pow]
        plain_strings = [self._print(e) for e in expr.args if not e.is_Pow]
        all_strings = sorted(plain_strings) + sorted(pow_strings)
        return '⋅'.join(all_strings)

    def _print_Pow(self, expr, rational=False):
        base = expr.args[0]
        exp = expr.args[1]
        if exp.is_Integer:
            exp_str = int_to_superscript(int(exp))
        else:
            exp_str = "^" + self._print(exp)
        return self._print(base) + exp_str

    def _print_Quantity(self, expr):
        # FIXME: sympy cannot handle the abbreviation of ml
        if str(expr) == "milliliter":
            return "ml"
        else:
            return str(expr.args[1])


class Quantity:
    def __init__(self, value: float, unit: Unit):
        self._value = value
        self._unit = unit

    @classmethod
    def parse(cls, s: str):
        m = re.match(r'(-?\d+(\.\d+)?)', s)
        if m:
            number_string = m.group(0)
            rest = s[len(number_string) :].strip()
            return cls(float(number_string), Unit(rest))
        else:
            raise ValueError(f"Unknown quantity {s}")

    def __eq__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        return self._value == other._value and self._unit == other._unit

    def __repr__(self):
        return f"{self._value} {self._unit}"

    @property
    def value(self):
        return self._value

    @property
    def unit(self):
        return self._unit
