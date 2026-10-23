# SPDX-FileCopyrightText: 2018-2026 The Pharmpy Contributors (see AUTHORS.rst)
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def test_sympy_is_sympy():
    import sympy

    from pharmpy.deps import sympy as sympy2

    assert sympy.Symbol is sympy2.Symbol


def test_sympy_dir():
    import sympy

    from pharmpy.deps import sympy as sympy2

    assert dir(sympy2) == dir(sympy)
