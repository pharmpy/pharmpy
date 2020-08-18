import sympy
from sympy import Symbol

from pharmpy import Model
from pharmpy.plugins.nonmem.advan import advan_equations


def S(x):
    return Symbol(x, real=True)


def test_pheno(pheno_path):
    model = Model(pheno_path)
    ode, ass = advan_equations(model)

    assert ass.symbol == S('F')
    assert ass.expression == S('A_1') / S('S1')

    t = S('t')
    assert ode.equation == sympy.Eq(sympy.Derivative(sympy.Function('A_1')(t), t),
                                    -S('CL') * sympy.Function('A_1')(t) / S('V'))
