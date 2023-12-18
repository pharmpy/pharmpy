import pytest

from pharmpy.internals.unicode import int_to_superscript, itemize_strings


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


@pytest.mark.parametrize(
    "x,expected",
    [
        (("cobolt",), "'cobolt'"),
        (("aa", "bb"), " one of 'aa' or 'bb'"),
        (("rock", "paper", "scissors"), " one of 'rock', 'paper' or 'scissors'"),
    ],
)
def test_itemize_strings(x, expected):
    assert itemize_strings(x) == expected
