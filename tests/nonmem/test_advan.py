import sympy
from sympy import Symbol

from pharmpy import Model
from pharmpy.plugins.nonmem.advan import compartmental_model


def S(x):
    return Symbol(x, real=True)


def test_pheno(pheno_path):
    model = Model(pheno_path)
    cm, ass = compartmental_model(model)

    assert ass.symbol == S('F')
    assert ass.expression == S('A_CENTRAL') / S('S1')
    assert cm.compartmental_matrix == sympy.Matrix([[-S('CL') / S('V')]])
    assert cm.amounts == sympy.Matrix([S('A_CENTRAL')])
    odes = cm.to_explicit_odes()
    assert len(odes) == 1
    assert str(odes[0]) == 'Eq(Derivative(A_CENTRAL(t), t), -CL*A_CENTRAL(t)/V)'
