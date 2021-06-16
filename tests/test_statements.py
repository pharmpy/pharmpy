import pytest
import sympy

import pharmpy.symbols
from pharmpy import Model
from pharmpy.modeling import explicit_odes
from pharmpy.statements import (
    Assignment,
    CompartmentalSystem,
    ExplicitODESystem,
    ModelStatements,
    ODESystem,
)


def S(x):
    return pharmpy.symbols.symbol(x)


def test_str(testdata):
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    assert str(s1) == 'KA := X + Y\n'
    s2 = Assignment(S('X2'), sympy.exp('X'))
    a = str(s2).split('\n')
    assert a[0].startswith(' ')
    assert len(a) == 3

    model = Model(testdata / 'nonmem' / 'pheno.mod')
    assert 'THETA(2)' in str(model.statements)
    assert 'THETA(2)' in repr(model.statements)


def test_subs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    statements.subs({'ETA(1)': 'ETAT1'})

    assert statements[5].expression == S('TVCL') * sympy.exp(S('ETAT1'))

    statements.subs({'TVCL': 'TVCLI'})

    assert statements[2].symbol == S('TVCLI')
    assert statements[5].expression == S('TVCLI') * sympy.exp(S('ETAT1'))

    assert statements.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}

    statements.subs({'V': 'V2'})

    assert statements.ode_system.free_symbols == {S('CL'), S('AMT'), S('t'), S('V2')}


def test_ode_system_base_class():
    odes = ODESystem()
    assert odes.free_symbols == set()
    assert odes.rhs_symbols == set()
    odes.subs({})
    assert odes == odes
    assert odes != CompartmentalSystem()
    assert str(odes) == 'ODESystem()'


def test_explicit_ode_system(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    explicit_odes(model)
    assert 'A_OUTPUT(0) = 0' in str(model.statements.ode_system)
    assert model.statements.ode_system.rhs_symbols == {S('t'), S('AMT'), S('CL'), S('V')}

    odes = ExplicitODESystem([sympy.Eq(S('x'), S('y'))], {})
    assert str(odes) == '{x = y'
    assert odes._repr_latex_() == '\\begin{cases} \\displaystyle x = y \\end{cases}'

    odes = ExplicitODESystem([sympy.Eq(S('x'), S('y')), sympy.Eq(S('z'), S('y'))], {})
    assert 'x = y' in str(odes)

    odes = ExplicitODESystem([sympy.Eq(S('x'), 1)], {})
    assert odes.free_symbols == {S('x')}


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
    assert str(statements.find_assignment('EPS(1)', is_symbol=False).symbol) == 'Y'

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
    s.remove_symbol_definitions([S('Z')], s1)
    assert s == ModelStatements([s4, s1])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s = ModelStatements([s1, s2, s3, s4])
    s.remove_symbol_definitions([S('CL')], s4)
    assert s == ModelStatements([s1, s4])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s5 = Assignment(S('KR'), S('CL'))
    s = ModelStatements([s1, s2, s3, s4, s5])
    s.remove_symbol_definitions([S('CL')], s4)
    assert s == ModelStatements([s1, s2, s3, s4, s5])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X'))
    s = ModelStatements([s1, s2, s3, s4])
    s.remove_symbol_definitions([S('CL'), S('K')], s4)
    assert s == ModelStatements([s4])


def test_remove_unused_parameters_and_rvs(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_real.mod')
    model.remove_unused_parameters_and_rvs()


def test_reassign():
    s1 = Assignment(S('G'), sympy.Integer(3))
    s2 = Assignment(S('M'), sympy.Integer(2))
    s3 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s4 = Assignment(S('KA'), S('X') + S('Y'))
    s = ModelStatements([s1, s2, s3, s4])
    s.reassign(S('M'), S('x') + S('y'))
    assert s == ModelStatements([s1, Assignment('M', S('x') + S('y')), s3, s4])

    s5 = Assignment('KA', S('KA') + S('Q') + 1)
    s = ModelStatements([s1, s2, s3, s4, s5])
    s.reassign(S('KA'), S('F'))
    assert s == ModelStatements([s1, s2, s3, Assignment('KA', S('F'))])


def test_find_compartment(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    comp = model.statements.ode_system.find_compartment('CENTRAL')
    assert comp.name == 'CENTRAL'
    comp = model.statements.ode_system.find_compartment('NOTINMODEL')
    assert comp is None


def test_find_output(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    model.statements.ode_system.add_compartment("NEW")
    with pytest.raises(ValueError):
        model.statements.ode_system.find_output()


def test_find_dosing(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.find_dosing().name == 'CENTRAL'
    model.statements.ode_system.find_dosing().dose = None
    with pytest.raises(ValueError):
        model.statements.ode_system.find_dosing()


def test_find_central(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_central().name == 'CENTRAL'
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.find_central().name == 'CENTRAL'


def test_find_depot(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None


def test_find_peripherals(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_peripherals() == []


def test_find_transit_compartments(testdata):
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert model.statements.ode_system.find_transit_compartments(model.statements) == []
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_transit_compartments(model.statements) == []
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_1transit.mod')
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 1
    assert transits[0].name == 'TRANS1'
    model = Model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 2
    assert transits[0].name == 'TRANS1'
    assert transits[1].name == 'TRANS2'


def test_add_before_odes(testdata):
    model = Model(testdata / 'nonmem' / 'minimal.mod')
    model.statements.add_before_odes(Assignment('CL', sympy.Integer(1)))
    model.update_source()
    assert str(model).split('\n')[6] == 'CL = 1'


def test_before_ode(pheno_path):
    model = Model(pheno_path)
    before_ode = model.statements.before_ode()
    assert before_ode[-1].symbol.name == 'S1'


def test_to_explicit_ode_system(pheno_path):
    model = Model(pheno_path)
    odes, ics = model.statements.ode_system.to_explicit_odes(skip_output=True)
    assert len(odes) == 1
    assert len(ics) == 1

    odes, ics = model.statements.ode_system.to_explicit_odes()
    assert len(odes) == 2
    assert len(ics) == 2


def test_print_custom(pheno_path):
    model = Model(pheno_path)
    s = model.statements.find_assignment('Y')

    s_str = s.print_custom(model.random_variables, None)

    assert s_str == 'Y = F + EPS(1)*W'


def test_repr_latex():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    latex = s1._repr_latex_()
    assert latex == r'$\displaystyle KA := \displaystyle X + Y$'


def test_repr_html():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    stats = ModelStatements([s1])
    html = stats._repr_html_()
    assert 'X + Y' in html


def test_dependencies(pheno_path):
    model = Model(pheno_path)
    depsy = model.statements.dependencies(S('Y'))
    assert depsy == {
        S('EPS(1)'),
        S('AMT'),
        S('THETA(1)'),
        S('t'),
        S('THETA(2)'),
        S('THETA(3)'),
        S('APGR'),
        S('WGT'),
        S('ETA(2)'),
        S('ETA(1)'),
    }
    depscl = model.statements.dependencies(S('CL'))
    assert depscl == {S('THETA(1)'), S('WGT'), S('ETA(1)')}
