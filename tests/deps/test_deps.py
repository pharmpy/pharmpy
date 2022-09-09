def test_sympy_is_sympy():
    import sympy

    from pharmpy.deps import sympy as sympy2

    assert sympy.Symbol is sympy2.Symbol


def test_sympy_dir():
    from pharmpy.deps import sympy

    assert 'Symbol' in dir(sympy)
