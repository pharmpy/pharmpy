from __future__ import annotations

import re
from typing import Literal, Optional, Union

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

    def is_compatible_with(self, other: Unit, molar_mass: Optional[float] = None) -> bool:
        """Check if this unit is compatible with (i.e. convertible to) another unit"""
        if molar_mass:
            unit1 = Unit(str(ureg.Quantity(1.0, self._units).to_base_units().units))
            unit1 = unit1._replace_unit_of_dimension(ureg.mol.dimensionality, ureg.gram)
            unit2 = Unit(str(ureg.Quantity(1.0, other._units).to_base_units().units))
            unit2 = unit2._replace_unit_of_dimension(ureg.mol.dimensionality, ureg.gram)
            is_compat = unit1.is_compatible_with(unit2)
        else:
            is_compat = self._units.is_compatible_with(other._units)
        return is_compat

    def get_dimensionality_string(self) -> str:
        """Get a human readable string with dimensionality of unit"""
        if self == Unit.unitless():
            return "1"
        s = str(ureg.get_dimensionality(self._units))
        s = s.replace("[", "").replace("]", "").replace(" ", "")
        return s

    def _replace_unit_of_dimension(self, dimension, unit) -> Unit:
        quant = pint.Quantity(1.0, self._units)
        items = quant.unit_items()
        new_unit = ureg("").units
        for unit_str, multiplicity in items:
            element = ureg(unit_str).units
            if element.dimensionality == dimension:
                new_unit *= unit**multiplicity
            else:
                new_unit *= element**multiplicity
        return Unit(new_unit)

    def replace_unit_of_dimension(self, replacement: Unit) -> Unit:
        """Replace the unit for one dimension"""
        return self._replace_unit_of_dimension(
            replacement._units.dimensionality, replacement._units
        )

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

    def convert_to(self, unit: Unit, molar_mass: Optional[Quantity] = None) -> Quantity:
        quant = ureg.Quantity(self._value, self._unit._units)
        if molar_mass:
            mw = ureg.Quantity(molar_mass._value, molar_mass._unit._units)
            new_quant = quant.to(unit._units, "chemistry", mw=mw)
        else:
            new_quant = quant.to(unit._units)
        return Quantity(new_quant.magnitude, unit)
