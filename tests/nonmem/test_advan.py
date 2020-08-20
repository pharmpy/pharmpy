import sympy
from sympy import Symbol

from pharmpy import Model
from pharmpy.plugins.nonmem.advan import compartmental_model


def S(x):
    return Symbol(x, real=True)


def test_pheno(pheno_path):
    model = Model(pheno_path)
    cm, ass = compartmental_model(model, 'ADVAN1', 'TRANS2')

    assert ass.symbol == S('F')
    assert ass.expression == S('A_CENTRAL') / S('S1')
    assert cm.compartmental_matrix == sympy.Matrix([[-S('CL') / S('V'), 0], [S('CL') / S('V'), 0]])
    assert cm.amounts == sympy.Matrix([S('A_CENTRAL'), S('A_OUTPUT')])
    odes, ics = cm.to_explicit_odes()
    assert len(odes) == 2
    assert str(odes[0]) == 'Eq(Derivative(A_CENTRAL(t), t), -CL*A_CENTRAL(t)/V)'
    assert str(odes[1]) == 'Eq(Derivative(A_OUTPUT(t), t), CL*A_CENTRAL(t)/V)'
    assert len(ics) == 2
    assert ics[sympy.Function('A_CENTRAL')(0)] == S('AMT')
    assert ics[sympy.Function('A_OUTPUT')(0)] == 0

    cm, ass = compartmental_model(model, 'ADVAN2', 'TRANS1')
    assert ass.symbol == S('F')
    assert ass.expression == S('A_CENTRAL') / S('S1')
    assert cm.compartmental_matrix == sympy.Matrix([[-S('KA'), 0, 0],
                                                    [S('KA'), -S('K'), 0],
                                                    [0, S('K'), 0]])
    assert cm.amounts == sympy.Matrix([S('A_DEPOT'), S('A_CENTRAL'), S('A_OUTPUT')])
    odes, ics = cm.to_explicit_odes()
    assert len(odes) == 3
    assert str(odes[0]) == 'Eq(Derivative(A_DEPOT(t), t), -KA*A_DEPOT(t))'
    assert str(odes[1]) == 'Eq(Derivative(A_CENTRAL(t), t), -K*A_CENTRAL(t) + KA*A_DEPOT(t))'
    assert str(odes[2]) == 'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))'
    assert len(ics) == 3
    assert ics[sympy.Function('A_DEPOT')(0)] == S('AMT')
    assert ics[sympy.Function('A_CENTRAL')(0)] == 0
    assert ics[sympy.Function('A_OUTPUT')(0)] == 0
