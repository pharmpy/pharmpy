import sympy


def real(name):
    return sympy.Symbol(name, real=True)


def subs(expr, substitutions):
    """Substitute symbols in sympy expression

        substitutions - dictionary from string or symbol to symbol or value
        sympification of symbols are assuming real symbols
    """
    d = dict()
    for key, value in substitutions.items():
        if isinstance(key, str):
            key = real(key)
        if isinstance(value, str):
            value = real(value)
        d[key] = value
    return expr.subs(d)


def sympify(expr_str):
    """Sympifies expression of type string with symbols set to real"""
    expr_sympy = sympy.sympify(expr_str)
    expr_symbols_str = [str(elem) for elem in expr_sympy.free_symbols]

    substitutions = dict(zip(expr_sympy.free_symbols, expr_symbols_str))

    expr_subs = subs(expr_sympy, substitutions)

    return expr_subs
