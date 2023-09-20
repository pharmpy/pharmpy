import pytest

from pharmpy.internals.unicode import int_to_superscript


@pytest.mark.parametrize(
    "x,expected",
    [
        (23, "²³"),
        (-1, "⁻¹"),
        (2, "²"),
    ],
)
def test_int_to_superscript(x, expected):
    assert int_to_superscript(x) == expected
