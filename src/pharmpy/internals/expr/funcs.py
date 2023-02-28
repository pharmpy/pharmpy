from pharmpy.deps import sympy

_smallz = 2.8e-103


class PHI(sympy.Function):
    pass


def PEXP(x):
    return sympy.Piecewise((sympy.exp(100), x > 100), (sympy.exp(x), True))


def PLOG(x):
    return sympy.Piecewise((sympy.log(_smallz), x < _smallz), (sympy.log(x), True))


def LOG10(x):
    return sympy.log(x, 10)


def PLOG10(x):
    return sympy.Piecewise((sympy.log(_smallz, 10), x < _smallz), (sympy.log(x, 10), True))


def PSQRT(x):
    return sympy.Piecewise((0, x < 0), (sympy.sqrt(x), True))


def INT(x):
    return sympy.sign(x) * sympy.floor(sympy.Abs(x))


def PDZ(x):
    return sympy.Piecewise((1 / _smallz, abs(x) < _smallz), (1 / x, True))


def PZR(x):
    return sympy.Piecewise((_smallz, abs(x) < _smallz), (x, True))


def PNP(x):
    return sympy.Piecewise((_smallz, x < _smallz), (x, True))


def PHE(x):
    return sympy.Piecewise((100, x > 100), (x, True))


def PNG(x):
    return sympy.Piecewise((0, x < 0), (x, True))
