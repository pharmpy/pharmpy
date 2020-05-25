import sympy

from pharmpy import Model


def S(x):
    return sympy.Symbol(x, real=True)


def test_subs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    statements.subs(S('ETA(1)'), S('ETAT1'))

    assert statements[5].expression == S('TVCL') * sympy.exp(S('ETAT1'))
