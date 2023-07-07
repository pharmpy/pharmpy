import pytest
import sympy

from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Infusion,
    Statements,
    output,
)
from pharmpy.modeling import add_effect_compartment, set_first_order_absorption


def S(x):
    return sympy.Symbol(x)


def test_statements_effect_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    model = add_effect_compartment(model, "baseline")

    with pytest.warns(UserWarning):
        print(model.statements)


def test_str(load_model_for_test, testdata):
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    assert str(s1) == 'KA = X + Y'
    s2 = Assignment(S('X2'), sympy.exp('X'))
    a = str(s2).split('\n')
    assert a[0].startswith(' ')
    assert len(a) == 2

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert 'TVV' in str(model.statements)
    assert 'TVV' in repr(model.statements)


def test_subs(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    s2 = statements.subs({'ETA_1': 'ETAT1'})

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

    assert str(statements.find_assignment('CL').expression) == 'TVCL*exp(ETA_1)'
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


def test_hash():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('KA'), S('X') + S('Z'))
    assert hash(s1) != hash(s2)

    b1 = Bolus(S('AMT'))
    b2 = Bolus(S('AMT'), admid=2)
    assert hash(b1) != hash(b2)

    i1 = Infusion("AMT", rate="R1")
    i2 = Infusion("AMT", rate="R2")
    assert hash(i1) != hash(i2)

    c1 = Compartment.create("DEPOT", lag_time="ALAG")
    c2 = Compartment.create("DEPOT")
    assert hash(c1) != hash(c2)
    assert hash(c1) != hash(output)

    st1 = Statements((s1, s2))
    st2 = Statements((s1,))
    assert hash(st1) != hash(st2)


def test_dict(load_model_for_test, testdata):
    ass1 = Assignment(S('KA'), S('X') + S('Y'))
    d = ass1.to_dict()
    assert d == {
        'class': 'Assignment',
        'symbol': "Symbol('KA')",
        'expression': "Add(Symbol('X'), Symbol('Y'))",
    }
    ass2 = Assignment.from_dict(d)
    assert ass1 == ass2

    dose1 = Bolus.create('AMT')
    d = dose1.to_dict()
    assert d == {'class': 'Bolus', 'amount': "Symbol('AMT')", 'admid': 1}
    dose2 = Bolus.from_dict(d)
    assert dose1 == dose2

    inf1 = Infusion.create('AMT', rate='R1')
    d = inf1.to_dict()
    assert d == {
        'class': 'Infusion',
        'amount': "Symbol('AMT')",
        'admid': 1,
        'rate': "Symbol('R1')",
        'duration': 'None',
    }
    inf2 = Infusion.from_dict(d)
    assert inf1 == inf2

    central = Compartment.create('CENTRAL', dose=dose1)
    d = central.to_dict()
    assert d == {
        'class': 'Compartment',
        'name': 'CENTRAL',
        'amount': "Symbol('A_CENTRAL')",
        'dose': {'class': 'Bolus', 'amount': "Symbol('AMT')", 'admid': 1},
        'input': 'Integer(0)',
        'lag_time': 'Integer(0)',
        'bioavailability': 'Integer(1)',
    }
    central2 = Compartment.from_dict(d)
    assert central == central2

    d = output.to_dict()
    assert d == {'class': 'Output'}
    output2 = output.__class__.from_dict(d)
    assert output == output2

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    odes = model.statements.ode_system
    d = odes.to_dict()
    assert d == {
        'class': 'CompartmentalSystem',
        'compartments': (
            {'class': 'Output'},
            {
                'class': 'Compartment',
                'name': 'CENTRAL',
                'amount': "Symbol('A_CENTRAL')",
                'dose': {'class': 'Bolus', 'amount': "Symbol('AMT')", 'admid': 1},
                'input': 'Integer(0)',
                'lag_time': 'Integer(0)',
                'bioavailability': 'Integer(1)',
            },
        ),
        'rates': [(1, 0, "Mul(Symbol('CL'), Pow(Symbol('V'), Integer(-1)))")],
        't': "Symbol('t')",
    }
    odes2 = CompartmentalSystem.from_dict(d)
    assert odes == odes2

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    d = model.statements.to_dict()
    assert d == {
        'statements': (
            {
                'class': 'Assignment',
                'symbol': "Symbol('Y')",
                'expression': "Add(Symbol('EPS_1'), Symbol('ETA_1'), Symbol('THETA_1'))",
            },
        )
    }
    stats2 = Statements.from_dict(d)
    assert model.statements == stats2


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


def test_dosing_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.dosing_compartment[0].name == 'CENTRAL'


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
    assert expr == sympy.Symbol("PTVCL") * sympy.Symbol("WGT") * sympy.exp(sympy.Symbol("ETA_1"))
    with pytest.raises(ValueError):
        model.statements.full_expression("Y")


def test_to_explicit_ode_system(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    cs = model.statements.ode_system
    assert len(cs.eqs) == 1

    assert cs.amounts == sympy.Matrix([sympy.Symbol('A_CENTRAL')])


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
    dose = Bolus.create('AMT')
    central = Compartment.create('CENTRAL', dose=dose)
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
        S('EPS_1'),
        S('AMT'),
        S('PTVCL'),
        S('t'),
        S('PTVV'),
        S('THETA_3'),
        S('APGR'),
        S('WGT'),
        S('ETA_2'),
        S('ETA_1'),
    }
    depscl = model.statements.dependencies(S('CL'))
    assert depscl == {S('PTVCL'), S('WGT'), S('ETA_1')}
    odes = model.statements.ode_system
    deps_odes = model.statements.dependencies(odes)
    assert deps_odes == {
        S('AMT'),
        S('APGR'),
        S('ETA_1'),
        S('ETA_2'),
        S('PTVCL'),
        S('PTVV'),
        S('THETA_3'),
        S('WGT'),
        S('t'),
    }
    with pytest.raises(KeyError):
        model.statements.dependencies("NONEXISTING")


def test_builder():
    cb = CompartmentalSystemBuilder()
    dose = Bolus('AMT')
    central = Compartment.create('CENTRAL', dose=dose)
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    depot = Compartment.create('DEPOT')
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cm = CompartmentalSystem(cb)
    assert cm.find_compartment('DEPOT').dose is None
    assert cm.central_compartment.dose == dose
    cb.move_dose(central, depot)
    cm2 = CompartmentalSystem(cb)
    assert cm2.find_compartment('DEPOT').dose == dose
    assert cm2.central_compartment.dose is None
    assert hash(cm) != hash(cm2)


def test_infusion_repr():
    inf = Infusion.create('AMT', rate='R1')
    assert repr(inf) == 'Infusion(AMT, admid=1, rate=R1)'
    inf = Infusion.create('AMT', duration='D1')
    assert repr(inf) == 'Infusion(AMT, admid=1, duration=D1)'
    inf = Infusion.create('AMT', admid=2, duration='D1')
    assert repr(inf) == 'Infusion(AMT, admid=2, duration=D1)'


def test_infusion_create():
    inf = Infusion.create('AMT', rate='R1')
    assert inf.rate == S('R1')

    with pytest.raises(ValueError):
        Infusion.create('AMT', rate='R1', duration='D1')

    with pytest.raises(ValueError):
        Infusion.create('AMT')


def test_compartment_repr():
    comp = Compartment.create("CENTRAL", lag_time='LT')
    assert repr(comp) == "Compartment(CENTRAL, amount=A_CENTRAL, lag_time=LT)"


def test_compartment_names(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.compartment_names == ['CENTRAL']


def test_assignment_create_numeric(load_model_for_test, testdata):
    with pytest.raises(AttributeError):
        Assignment('X', 1).free_symbols
    assert Assignment.create('X', 1).free_symbols
    with pytest.raises(AttributeError):
        Assignment('X', 1.0).free_symbols
    assert Assignment.create('X', 1.0).free_symbols


def test_multi_dose_comp_order(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_first_order_absorption(model)

    ode = model.statements.ode_system
    cb = CompartmentalSystemBuilder(ode)
    cb.set_dose(cb.find_compartment("CENTRAL"), cb.find_compartment("DEPOT").dose)
    ode = CompartmentalSystem(cb)
    assert ode.dosing_compartment[0].name == "DEPOT"
