# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import Literal

import pytest

from pharmpy.internals.fn.type import _match, check_list


def test_check_list():
    with pytest.raises(ValueError):
        check_list("my", "a", ("b", "c", "d"))

    check_list("my", "a", ("a", "b", "c"))


def test_match():
    assert _match(dict[str, int], {'a': 1})
    assert not _match(dict[str, int], {'a': '1'})
    assert _match(dict[Literal['a', 'b'], int], {'b': 1})
    assert _match(dict[Literal['a', 'b'], str], {'b': 'c'})
    assert not _match(dict[Literal['a', 'b'], int], {'c': 1})
    assert _match(dict[int, int], {1: 2})
    assert not _match(dict[int, int], {1: '2'})
