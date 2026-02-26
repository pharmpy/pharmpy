from __future__ import annotations

import re
from typing import Literal, Union

from pharmpy.deps import pint

ureg = pint.UnitRegistry()
ureg.define("l = liter = L")


class Unit:
    def __init__(self, source: Union[Unit, str, Literal[1], pint.Unit]):
        if isinstance(source, Unit):
            self._units = source._units
        elif isinstance(source, pint.Unit):
            self._units = source
        else:
            if source == "1" or source == 1:
                self._units = ureg.dimensionless
            else:
                try:
                    self._units = ureg(source).units  # pyright: ignore [reportArgumentType]
                except pint.errors.UndefinedUnitError:
                    raise ValueError(f"Unknown unit {source}")

    def unicode(self) -> str:
        if self._units == ureg.dimensionless:
            s = "1"
        else:
            s = f"{self._units:~P}"
            # Conversion needed because of https://github.com/hgrecco/pint/issues/2272
            s = s.replace("·", "⋅")
        return s

    def serialize(self) -> str:
        if self._units == ureg.dimensionless:
            s = "1"
        else:
            s = f"{self._units:~P}"
        return s

    @classmethod
    def deserialize(cls, s) -> Unit:
        return cls(s)

    @classmethod
    def unitless(cls) -> Unit:
        return cls("1")

    def is_compatible_with(self, other: Unit) -> bool:
        """Check if this unit is compatible with (i.e. convertible to) another unit"""
        return self._units.is_compatible_with(other._units)

    def get_dimensionality_string(self) -> str:
        """Get a human readable string with dimensionality of unit"""
        if self == Unit.unitless():
            return "1"
        s = str(ureg.get_dimensionality(self._units))
        s = s.replace("[", "").replace("]", "").replace(" ", "")
        return s

    def __mul__(self, other: Unit) -> Unit:
        return Unit(self._units * other._units)

    def __truediv__(self, other: Unit) -> Unit:
        return Unit(self._units / other._units)  # pyright: ignore [reportArgumentType]

    def __pow__(self, n: int) -> Unit:
        return Unit(self._units**n)  # pyright: ignore [reportArgumentType]

    def __eq__(self, other):
        if not isinstance(other, Unit):
            return NotImplemented
        return self._units == other._units

    def __hash__(self):
        return hash(self._units)

    def __repr__(self):
        return self.unicode()


# Type hint for public functions taking an expression as input
TUnit = str | Unit


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
    def value(self) -> float:
        return self._value

    @property
    def unit(self) -> Unit:
        return self._unit

    def convert_to(self, unit: Unit) -> Quantity:
        quant = pint.Quantity(self._value, self._unit._units)
        new_quant = quant.to(unit._units)
        return Quantity(new_quant.magnitude, unit)
