import sympy

from pharmpy import Model
from pharmpy.statements import Compartment, Elimination, IVAbsorption


def S(x):
    return sympy.Symbol(x, real=True)


def test_subs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    statements.subs(S('ETA(1)'), S('ETAT1'))

    assert statements[5].expression == S('TVCL') * sympy.exp(S('ETAT1'))


def test_Compartment():
    absorption = IVAbsorption('AMT')
    CL = S('CL')
    V = S('V')
    elimination = Elimination(CL / V)
    comp = Compartment('central', absorption, elimination)
    assert comp


def test_eq_assignment(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements
    statement_btime = statements[0]
    statement_tad = statements[1]

    assert statement_btime == statement_btime
    assert statement_btime != statement_tad


def test_eq_modelstatements(testdata):
    model_min = Model(testdata / 'nonmem' / 'minimal.mod')
    model_pheno = Model(testdata / 'nonmem' / 'pheno_real.mod')

    assert model_min.statements == model_min.statements
    assert model_pheno.statements == model_pheno.statements
    assert model_min.statements != model_pheno.statements
