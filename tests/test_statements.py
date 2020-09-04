import sympy

from pharmpy import Model
from pharmpy.modeling import explicit_odes
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import real


def S(x):
    return real(x)


def test_subs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    statements.subs({'ETA(1)': 'ETAT1'})

    assert statements[5].expression == S('TVCL') * sympy.exp(S('ETAT1'))

    statements.subs({'TVCL': 'TVCLI'})

    assert statements[2].symbol == S('TVCLI')
    assert statements[5].expression == S('TVCLI') * sympy.exp(S('ETAT1'))


def test_ode_free_symbols(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')

    assert model.statements.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}

    explicit_odes(model)
    odes = model.statements.ode_system
    assert odes.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}


def test_find_assignment(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    assert str(statements.find_assignment('CL').expression) == 'TVCL*exp(ETA(1))'
    assert str(statements.find_assignment('S1').expression) == 'V'

    statements.append(Assignment(S('CL'), S('TVCL') + S('V')))

    assert str(statements.find_assignment('CL').expression) == 'TVCL + V'


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


def test_remove_symbol_definition():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s3 = Assignment(S('M'), sympy.Integer(2))
    s4 = Assignment(S('G'), sympy.Integer(3))
    s = ModelStatements([s4, s3, s2, s1])
    s.remove_symbol_definition(S('Z'), s1)
    assert s == ModelStatements([s4, s1])


def test_remove_unused_parameters_and_rvs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    model.remove_unused_parameters_and_rvs()
