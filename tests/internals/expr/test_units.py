import pytest

from pharmpy.internals.expr.units import parse, unit_string


@pytest.mark.parametrize(
    "x,expected",
    [("kg", "kg"), ("m/s", "m⋅s⁻¹"), ("km/s", "km⋅s⁻¹"), ("mg/mL", "mg⋅ml⁻¹")],
)
def test_unit_string(x, expected):
    assert unit_string(parse(x)) == expected
