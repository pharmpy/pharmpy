import pytest
import sympy

from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    ExplicitODESystem,
    Infusion,
    Statements,
)


def S(x):
    return sympy.Symbol(x)


def test_str(load_model_for_test, testdata):
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    assert str(s1) == 'KA = X + Y'
    s2 = Assignment(S('X2'), sympy.exp('X'))
    a = str(s2).split('\n')
    assert a[0].startswith(' ')
    assert len(a) == 2

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert 'THETA(2)' in str(model.statements)
    assert 'THETA(2)' in repr(model.statements)


def test_subs(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    s2 = statements.subs({'ETA(1)': 'ETAT1'})

    assert s2[5].expression == S('TVCL') * sympy.exp(S('ETAT1'))

    s3 = s2.subs({'TVCL': 'TVCLI'})

    assert s3[2].symbol == S('TVCLI')
    assert s3[5].expression == S('TVCLI') * sympy.exp(S('ETAT1'))

    assert s3.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}

    s4 = s3.subs({'V': 'V2'})

    assert s4.ode_system.free_symbols == {S('CL'), S('AMT'), S('t'), S('V2')}


def test_ode_free_symbols(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert model.statements.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}


def test_find_assignment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    assert str(statements.find_assignment('CL').expression) == 'TVCL*exp(ETA(1))'
    assert str(statements.find_assignment('S1').expression) == 'V'

    statements = statements + Assignment(S('CL'), S('TVCL') + S('V'))

    assert str(statements.find_assignment('CL').expression) == 'TVCL + V'


def test_eq_assignment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements
    statement_btime = statements[0]
    statement_tad = statements[1]

    assert statement_btime == statement_btime
    assert statement_btime != statement_tad


def test_eq_modelstatements(load_model_for_test, testdata):
    model_min = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model_pheno = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert model_min.statements == model_min.statements
    assert model_pheno.statements == model_pheno.statements
    assert model_min.statements != model_pheno.statements

    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s3 = Assignment(S('M'), sympy.Integer(2))
    s4 = Assignment(S('G'), sympy.Integer(3))
    s1 = Statements([s2, s1])
    s2 = Statements([s3, s4])
    assert s1 != s2


def test_add():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s3 = Assignment(S('M'), sympy.Integer(2))
    s = Statements([s1, s2])
    new = (s3,) + s
    assert len(new) == 3


def test_remove_symbol_definition():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s3 = Assignment(S('M'), sympy.Integer(2))
    s4 = Assignment(S('G'), sympy.Integer(3))
    s = Statements([s4, s3, s2, s1])
    ns = s.remove_symbol_definitions([S('Z')], s1)
    assert ns == Statements([s4, s1])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s = Statements([s1, s2, s3, s4])
    ns = s.remove_symbol_definitions([S('CL')], s4)
    assert ns == Statements([s1, s4])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s5 = Assignment(S('KR'), S('CL'))
    s = Statements([s1, s2, s3, s4, s5])
    ns = s.remove_symbol_definitions([S('CL')], s4)
    assert ns == Statements([s1, s2, s3, s4, s5])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X'))
    s = Statements([s1, s2, s3, s4])
    ns = s.remove_symbol_definitions([S('CL'), S('K')], s4)
    assert ns == Statements([s4])

    s1 = Assignment(S('K'), sympy.Integer(16))
    s2 = Assignment(S('CL'), sympy.Integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X'))
    s5 = Assignment(S('P'), S('K'))
    s = Statements([s1, s2, s3, s4, s5])
    ns = s.remove_symbol_definitions([S('CL'), S('K')], s4)
    assert ns == Statements([s1, s4, s5])


def test_reassign():
    s1 = Assignment(S('G'), sympy.Integer(3))
    s2 = Assignment(S('M'), sympy.Integer(2))
    s3 = Assignment(S('Z'), sympy.Integer(23) + S('M'))
    s4 = Assignment(S('KA'), S('X') + S('Y'))
    s = Statements([s1, s2, s3, s4])
    snew = s.reassign(S('M'), S('x') + S('y'))
    assert snew == Statements([s1, Assignment(S('M'), S('x') + S('y')), s3, s4])

    s5 = Assignment(S('KA'), S('KA') + S('Q') + 1)
    s = Statements([s1, s2, s3, s4, s5])
    snew = s.reassign(S('KA'), S('F'))
    assert snew == Statements([s1, s2, s3, Assignment(S('KA'), S('F'))])
    snew = s.reassign('KA', 'F')
    assert snew == Statements([s1, s2, s3, Assignment(S('KA'), S('F'))])


def test_find_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    comp = model.statements.ode_system.find_compartment('CENTRAL')
    assert comp.name == 'CENTRAL'
    comp = model.statements.ode_system.find_compartment('NOTINMODEL')
    assert comp is None


def test_output_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    cb = CompartmentalSystemBuilder(model.statements.ode_system)
    cb.add_compartment("NEW")
    cm = CompartmentalSystem(cb)
    with pytest.raises(ValueError):
        cm.output_compartment


def test_dosing_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.dosing_compartment.name == 'CENTRAL'


def test_central_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.central_compartment.name == 'CENTRAL'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.central_compartment.name == 'CENTRAL'


def test_find_depot(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None


def test_peripheral_compartments(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.peripheral_compartments == []


def test_find_transit_compartments(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert model.statements.ode_system.find_transit_compartments(model.statements) == []
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_transit_compartments(model.statements) == []
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_1transit.mod')
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 1
    assert transits[0].name == 'TRANS1'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 2
    assert transits[0].name == 'TRANS1'
    assert transits[1].name == 'TRANS2'


def test_before_odes(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    before_ode = model.statements.before_odes
    assert before_ode[-1].symbol.name == 'S1'


def test_full_expression(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    expr = model.statements.before_odes.full_expression("CL")
    assert expr == sympy.Symbol("THETA(1)") * sympy.Symbol("WGT") * sympy.exp(
        sympy.Symbol("ETA(1)")
    )
    with pytest.raises(ValueError):
        model.statements.full_expression("Y")


def test_to_explicit_ode_system(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    exodes = model.statements.ode_system.to_explicit_system(skip_output=True)
    odes, ics = exodes.odes, exodes.ics
    assert len(odes) == 1
    assert len(ics) == 1

    exodes = model.statements.ode_system.to_explicit_system()
    odes, ics = exodes.odes, exodes.ics
    assert len(odes) == 2
    assert len(ics) == 2

    assert exodes.amounts == sympy.Matrix([sympy.Symbol('A_CENTRAL'), sympy.Symbol('A_OUTPUT')])

    newstats = model.statements.to_explicit_system()
    assert isinstance(newstats.ode_system, ExplicitODESystem)

    back = model.statements.to_compartmental_system()
    assert isinstance(back.ode_system, CompartmentalSystem)
    assert len(newstats) == len(back)


def test_repr_latex():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    latex = s1._repr_latex_()
    assert latex == r'$KA = X + Y$'


def test_repr_html():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    stats = Statements([s1])
    html = stats._repr_html_()
    assert 'X + Y' in html

    cb = CompartmentalSystemBuilder()
    dose = Bolus('AMT')
    central = Compartment('CENTRAL', dose)
    output = Compartment('OUTPUT')
    cb.add_compartment(central)
    cb.add_compartment(output)
    cb.add_flow(central, output, S('K'))
    cs = CompartmentalSystem(cb)
    stats = Statements([cs])
    assert type(stats._repr_html_()) == str


def test_direct_dependencies(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    odes = model.statements.ode_system
    deps = model.statements.direct_dependencies(odes)
    assert deps[0].symbol.name == "CL"
    assert deps[1].symbol.name == "V"


def test_dependencies(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
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
    odes = model.statements.ode_system
    deps_odes = model.statements.dependencies(odes)
    assert deps_odes == {
        S('AMT'),
        S('APGR'),
        S('ETA(1)'),
        S('ETA(2)'),
        S('THETA(1)'),
        S('THETA(2)'),
        S('THETA(3)'),
        S('WGT'),
        S('t'),
    }
    with pytest.raises(KeyError):
        model.statements.dependencies("NONEXISTING")


def test_builder():
    cb = CompartmentalSystemBuilder()
    dose = Bolus('AMT')
    central = Compartment('CENTRAL', dose)
    output = Compartment('OUTPUT')
    cb.add_compartment(central)
    cb.add_compartment(output)
    cb.add_flow(central, output, S('K'))
    depot = Compartment('DEPOT')
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cm = CompartmentalSystem(cb)
    assert cm.find_compartment('DEPOT').dose is None
    assert cm.central_compartment.dose == dose
    cb.move_dose(central, depot)
    cm = CompartmentalSystem(cb)
    assert cm.find_compartment('DEPOT').dose == dose
    assert cm.central_compartment.dose is None


def test_infusion_repr():
    inf = Infusion('AMT', rate='R1')
    assert repr(inf) == 'Infusion(AMT, rate=R1)'
    inf = Infusion('AMT', duration='D1')
    assert repr(inf) == 'Infusion(AMT, duration=D1)'


def test_infusion_create():
    inf = Infusion.create('AMT', rate='R1')
    assert inf.rate == S('R1')

    with pytest.raises(ValueError):
        Infusion.create('AMT', rate='R1', duration='D1')

    with pytest.raises(ValueError):
        Infusion.create('AMT')


def test_compartment_repr():
    comp = Compartment("CENTRAL", lag_time='LT')
    assert repr(comp) == "Compartment(CENTRAL, lag_time=LT)"


def test_compartment_names(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.compartment_names == ['CENTRAL', 'OUTPUT']
