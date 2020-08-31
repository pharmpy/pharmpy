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
        d[key] = value
    return expr.subs(d)
