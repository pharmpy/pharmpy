import pytest

from pharmpy.basic.unit import Quantity, Unit


def test_init():
    unit1 = Unit('ml')
    assert repr(unit1) == 'mL'

    unit2 = Unit(unit1)
    assert unit1 == unit2

    with pytest.raises(ValueError):
        Unit("myownunit")


def test_unitless():
    assert Unit.unitless() == Unit(1)


def test_serialize():
    unit = Unit('m')
    unit_ser = unit.serialize()
    assert unit_ser == "m"
    assert Unit.deserialize(unit_ser) == unit
    assert Unit(1).serialize() == "1"


@pytest.mark.parametrize(
    'expr, unicode_ref',
    [('g', 'g'), ('g*h', 'g⋅h'), ('h^-1', '1/h'), ('ml', 'mL'), ('ug', 'µg'), ('µl', 'µL')],
)
def test_unicode(expr, unicode_ref):
    assert Unit(expr).unicode() == unicode_ref


@pytest.mark.parametrize(
    'unit1,unit2,correct',
    [('mg', 'mg', True), ('g', 'mg', True), ('g/L', 'g/mL', True), ('h*m', 'h', False)],
)
def test_is_convertible_to(unit1, unit2, correct):
    assert Unit(unit1).is_compatible_with(Unit(unit2)) is correct


def test_hash():
    x = Unit("kg")
    assert hash(x) == hash(Unit("kg"))


@pytest.mark.parametrize(
    'unit,correct',
    [
        ('mg', 'mass'),
        ('kg', 'mass'),
        ('1', '1'),
        ('m/s', 'length/time'),
    ],
)
def test_get_dimensionality_string(unit, correct):
    assert Unit(unit).get_dimensionality_string() == correct


@pytest.mark.parametrize(
    'unit,replacement,correct',
    [
        ('mg', 'mg', 'mg'),
        ('mg/L', 'ug', 'ug/L'),
    ],
)
def test_replace_unit_of_dimension(unit, replacement, correct):
    orig = Unit(unit)
    new = orig.replace_unit_of_dimension(Unit(replacement))
    assert new == Unit(correct)


def test_quantity():
    x = Quantity(2.5, Unit("mg"))
    assert x.value == 2.5
    assert x.unit == Unit("mg")
    assert repr(x) == "2.5 mg"

    y = Quantity(1.5, Unit("mg"))
    assert x != y
    assert x == x
    assert x != Unit("mg")

    z = Quantity.parse("-2.5mg")
    assert z.value == -2.5
    assert z.unit == Unit("mg")

    with pytest.raises(ValueError):
        Quantity.parse("mg")

    w = x.convert_to(Unit("g"))
    assert w.value == 0.0025
    assert w.unit == Unit("g")

    x = Quantity(1.0, Unit("mg/L"))
    mm = Quantity(163.0, Unit("g/mol"))
    new = x.convert_to(Unit("M"), molar_mass=mm)
    assert new.unit == Unit("M")
