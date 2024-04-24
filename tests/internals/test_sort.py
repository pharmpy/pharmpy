import pytest

from pharmpy.internals.sort import sort_alphanum


@pytest.mark.parametrize(
    'array,correct',
    (
        (("a", "b", "c"), ["a", "b", "c"]),
        (("a2", "a1", "a3"), ["a1", "a2", "a3"]),
        (("a2", "a1", "a3"), ["a1", "a2", "a3"]),
        (("a10", "a9"), ["a9", "a10"]),
        (("a1m", "a1k"), ["a1k", "a1m"]),
    ),
)
def test_sort_alphanum(array, correct):
    assert sort_alphanum(array) == correct
