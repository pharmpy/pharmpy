import pytest

from pharmpy.internals.fn.type import check_list


def test_check_list():
    with pytest.raises(ValueError):
        check_list("my", "a", ("b", "c", "d"))

    check_list("my", "a", ("a", "b", "c"))
