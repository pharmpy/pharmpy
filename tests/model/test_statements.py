import pytest

from pharmpy.basic import Expr, Matrix
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
from pharmpy.model.statements import Output, to_compartmental_system
from pharmpy.modeling import add_effect_compartment, set_first_order_absorption


def S(x):
    return Expr.symbol(x)


def test_statements_effect_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    model = add_effect_compartment(model, "linear")

    with pytest.warns(UserWarning):
        str(model.statements)


def test_str(load_model_for_test, testdata):
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    assert str(s1) == 'KA = X + Y'
    s2 = Assignment(S('X2'), Expr.symbol('X').exp())
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

    assert s2[5].expression == S('TVCL') * (S('ETAT1').exp())

    s3 = s2.subs({'TVCL': 'TVCLI'})

    assert s3[2].symbol == S('TVCLI')
    assert s3[5].expression == S('TVCLI') * (S('ETAT1').exp())

    assert s3.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}

    s4 = s3.subs({'V': 'V2'})

    assert s4.ode_system.free_symbols == {S('CL'), S('AMT'), S('t'), S('V2')}


def test_ode_free_symbols(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert model.statements.ode_system.free_symbols == {S('V'), S('CL'), S('AMT'), S('t')}


def test_lhs_symbols(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    assert model.statements.lhs_symbols == {
        S('F'),
        Expr.function("A_CENTRAL", "t"),
        S("TVV"),
        S("CL"),
        S("Y"),
        S("VC"),
        S("V"),
        S("TVCL"),
        S("S1"),
    }


def test_find_assignment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    assert str(statements.find_assignment('CL').expression) == 'TVCL*exp(ETA_1)'
    assert str(statements.find_assignment('S1').expression) == 'V'

    statements = statements + Assignment.create(S('CL'), S('TVCL') + S('V'))

    assert str(statements.find_assignment('CL').expression) == 'TVCL + V'


def test_get_assignment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements

    assert str(statements.get_assignment('CL').expression) == 'TVCL*exp(ETA_1)'
    assert str(statements.get_assignment('S1').expression) == 'V'
    with pytest.raises(ValueError):
        statements.get_assignment("X")


def test_eq_assignment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    statements = model.statements
    statement_btime = statements[0]
    statement_tad = statements[1]

    assert statement_btime == statement_btime
    assert statement_btime != statement_tad


def test_eq_statements(load_model_for_test, testdata):
    model_min = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model_pheno = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    assert model_min.statements == model_min.statements
    assert model_pheno.statements == model_pheno.statements
    assert model_min.statements != model_pheno.statements

    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), Expr.integer(23) + S('M'))
    s3 = Assignment(S('M'), Expr.integer(2))
    s4 = Assignment(S('G'), Expr.integer(3))
    s1 = Statements([s2, s1])
    s2 = Statements([s3, s4])
    assert s1 != s2


def test_hash():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('KA'), S('X') + S('Z'))
    assert hash(s1) != hash(s2)

    b1 = Bolus(Expr.symbol('AMT'))
    b2 = Bolus(Expr.symbol('AMT'), admid=2)
    assert hash(b1) != hash(b2)

    i1 = Infusion.create("AMT", rate="R1")
    i2 = Infusion.create("AMT", rate="R2")
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
        'duration': None,
    }
    inf2 = Infusion.from_dict(d)
    assert inf1 == inf2

    central = Compartment.create('CENTRAL', doses=(dose1,))
    d = central.to_dict()
    assert d == {
        'class': 'Compartment',
        'name': 'CENTRAL',
        'amount': "Function('A_CENTRAL')(Symbol('t'))",
        'doses': ({'class': 'Bolus', 'amount': "Symbol('AMT')", 'admid': 1},),
        'input': 'Integer(0)',
        'lag_time': 'Integer(0)',
        'bioavailability': 'Integer(1)',
    }
    central2 = Compartment.from_dict(d)
    assert central == central2

    central = Compartment.create('CENTRAL', doses=tuple())
    d = central.to_dict()
    assert d == {
        'class': 'Compartment',
        'name': 'CENTRAL',
        'amount': "Function('A_CENTRAL')(Symbol('t'))",
        'doses': None,
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
                'amount': "Function('A_CENTRAL')(Symbol('t'))",
                'doses': ({'class': 'Bolus', 'amount': "Symbol('AMT')", 'admid': 1},),
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

    cb = CompartmentalSystemBuilder()
    dose = Infusion.create('AMT', rate='R1')
    central = Compartment.create('CENTRAL', doses=(dose,))
    cb.add_compartment(central)
    cb.add_compartment(output)
    cb.add_flow(central, output, S('K'))
    odes = CompartmentalSystem(cb)
    d = odes.to_dict()
    assert d == {
        'class': 'CompartmentalSystem',
        'compartments': (
            {'class': 'Output'},
            {
                'class': 'Compartment',
                'name': 'CENTRAL',
                'amount': "Function('A_CENTRAL')(Symbol('t'))",
                'doses': (
                    {
                        'admid': 1,
                        'amount': "Symbol('AMT')",
                        'class': 'Infusion',
                        'duration': None,
                        'rate': "Symbol('R1')",
                    },
                ),
                'input': 'Integer(0)',
                'lag_time': 'Integer(0)',
                'bioavailability': 'Integer(1)',
            },
        ),
        'rates': [(1, 0, "Symbol('K')")],
        't': "Symbol('t')",
    }
    odes2 = CompartmentalSystem.from_dict(d)
    assert odes == odes2


def test_add():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), Expr.integer(23) + S('M'))
    new = s1 + s2
    assert len(new) == 2
    s3 = Assignment(S('M'), Expr.integer(2))
    s = Statements([s1, s2])
    new = s + s3
    assert len(new) == 3
    new = s3 + s
    assert len(new) == 3
    new = (s3,) + s
    assert len(new) == 3
    new = s + (s3,)
    assert len(new) == 3
    new = s1 + (s2,)
    assert len(new) == 2
    new = (s1,) + s2
    assert len(new) == 2
    new = s + s
    assert len(new) == 4

    with pytest.raises(TypeError):
        s1 + 1
    with pytest.raises(TypeError):
        1 + s1

    with pytest.raises(TypeError):
        s + 1
    with pytest.raises(TypeError):
        1 + s


def test_assignment_create():
    s1 = Assignment.create(S('KA'), S('X') + S('Y'))
    assert s1.symbol == S('KA')
    assert s1.expression == S('X') + S('Y')
    s2 = Assignment.create('KA', 'X+Y')
    assert s2.symbol == S('KA')
    assert s2.expression == S('X') + S('Y')
    assert s1 == s2
    with pytest.raises(TypeError):
        Assignment.create(S('X') + S('Y'), S('KA'))
    with pytest.raises(TypeError):
        Assignment.create(s1, s2)


def test_compartment_create():
    dose = Bolus.create('AMT')
    comp = Compartment.create('CENTRAL', doses=(dose,))
    assert comp.name == 'CENTRAL'
    assert comp.amount == Expr.function('A_CENTRAL', 't')

    comp = Compartment.create('CENTRAL', doses=[dose])
    assert isinstance(comp.doses, tuple)

    with pytest.raises(TypeError):
        Compartment.create(1)

    with pytest.raises(TypeError):
        Compartment.create('CENTRAL', doses=dose)

    with pytest.raises(TypeError):
        Compartment.create('CENTRAL', doses=[1])


def test_compartmental_system_create(load_example_model_for_test):
    odes = load_example_model_for_test('pheno').statements.ode_system

    cb = CompartmentalSystemBuilder(odes)
    cs = CompartmentalSystem.create(cb)

    assert cs == odes

    with pytest.raises(TypeError):
        CompartmentalSystem.create(None)
    with pytest.raises(TypeError):
        CompartmentalSystem.create(1)
    with pytest.raises(TypeError):
        CompartmentalSystem.create(cb, 1)


def test_statements_create():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    sset1 = Statements.create([s1])
    assert len(sset1) == 1

    s2 = Assignment(S('A'), S('B'))
    sset2 = Statements.create([s1, s2])
    assert len(sset2) == 2

    sset3 = Statements.create(sset1)
    assert sset1 == sset3

    sset4 = Statements.create(None)
    assert len(sset4) == 0

    with pytest.raises(TypeError):
        Statements.create('x')

    with pytest.raises(TypeError):
        Statements.create((s1, 'x'))

    with pytest.raises(TypeError):
        Statements.create(1)


def test_remove_symbol_definition():
    s1 = Assignment(S('KA'), S('X') + S('Y'))
    s2 = Assignment(S('Z'), Expr.integer(23) + S('M'))
    s3 = Assignment(S('M'), Expr.integer(2))
    s4 = Assignment(S('G'), Expr.integer(3))
    s = Statements([s4, s3, s2, s1])
    ns = s.remove_symbol_definitions([Expr.symbol('Z')], s1)
    assert ns == Statements([s4, s1])

    s1 = Assignment(S('K'), Expr.integer(16))
    s2 = Assignment(S('CL'), Expr.integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s = Statements([s1, s2, s3, s4])
    ns = s.remove_symbol_definitions([Expr.symbol('CL')], s4)
    assert ns == Statements([s1, s4])

    s1 = Assignment(S('K'), Expr.integer(16))
    s2 = Assignment(S('CL'), Expr.integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X') + S('K'))
    s5 = Assignment(S('KR'), S('CL'))
    s = Statements([s1, s2, s3, s4, s5])
    ns = s.remove_symbol_definitions([Expr.symbol('CL')], s4)
    assert ns == Statements([s1, s2, s3, s4, s5])

    s1 = Assignment(S('K'), Expr.integer(16))
    s2 = Assignment(S('CL'), Expr.integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X'))
    s = Statements([s1, s2, s3, s4])
    ns = s.remove_symbol_definitions([Expr.symbol('CL'), Expr.symbol('K')], s4)
    assert ns == Statements([s4])

    s1 = Assignment(S('K'), Expr.integer(16))
    s2 = Assignment(S('CL'), Expr.integer(23))
    s3 = Assignment(S('CL'), S('CL') + S('K'))
    s4 = Assignment(S('G'), S('X'))
    s5 = Assignment(S('P'), S('K'))
    s = Statements([s1, s2, s3, s4, s5])
    ns = s.remove_symbol_definitions([Expr.symbol('CL'), Expr.symbol('K')], s4)
    assert ns == Statements([s1, s4, s5])


def test_reassign():
    s1 = Assignment(S('G'), Expr.integer(3))
    s2 = Assignment(S('M'), Expr.integer(2))
    s3 = Assignment(S('Z'), Expr.integer(23) + S('M'))
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
    odes = model.statements.ode_system
    comp = odes.find_compartment('CENTRAL')
    assert comp.name == 'CENTRAL'
    comp = odes.find_compartment('NOTINMODEL')
    assert comp is None

    comp = odes.find_compartment_or_raise('CENTRAL')
    assert comp.name == 'CENTRAL'
    with pytest.raises(ValueError):
        odes.find_compartment_or_raise('NOTINMODEL')

    cb = CompartmentalSystemBuilder(odes)
    comp = cb.find_compartment('CENTRAL')
    assert comp.name == 'CENTRAL'
    comp = cb.find_compartment('NOTINMODEL')
    assert comp is None


def test_dosing_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    cs = model.statements.ode_system
    assert len(cs.dosing_compartments) == 1
    assert cs.dosing_compartments[0].name == 'CENTRAL'

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    odes = model.statements.ode_system
    cb = CompartmentalSystemBuilder(odes)
    central = odes.central_compartment
    dose = Bolus(Expr.symbol('AMT'))
    cb.set_dose(central, dose=dose)
    cs = CompartmentalSystem(cb)

    assert len(cs.dosing_compartments) == 2
    assert cs.dosing_compartments[-1].name == 'CENTRAL'

    peripheral = Compartment.create('PERIPHERAL')
    cb.add_compartment(peripheral)
    cb.add_flow(central, peripheral, S('X'))
    cb.add_flow(peripheral, central, S('X'))
    cb.set_dose(peripheral, dose=dose)
    cs = CompartmentalSystem(cb)
    assert len(cs.dosing_compartments) == 3
    assert cs.dosing_compartments[-1].name == 'CENTRAL'


def test_central_compartment(load_model_for_test, load_example_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.central_compartment.name == 'CENTRAL'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.central_compartment.name == 'CENTRAL'

    odes = model.statements.ode_system
    cb = CompartmentalSystemBuilder(odes)
    comp = Compartment.create('METABOLITE')
    central = odes.central_compartment
    cb.add_compartment(comp)
    cb.add_flow(central, comp, 'X')
    cb.remove_flow(central, output)
    cb.add_flow(comp, output, 'Y')
    cs = CompartmentalSystem(cb)
    assert cs.central_compartment.name == 'CENTRAL'

    cb = CompartmentalSystemBuilder(odes)
    central = odes.central_compartment
    cb.remove_compartment(central)
    cb.add_compartment(comp)
    cb.add_flow(comp, output, 'Y')
    cs = CompartmentalSystem(cb)

    with pytest.raises(ValueError):
        cs.central_compartment

    cb = CompartmentalSystemBuilder(odes)
    central = odes.central_compartment
    cb.remove_flow(central, output)
    cs = CompartmentalSystem(cb)

    with pytest.raises(ValueError):
        cs.central_compartment


def test_find_depot(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    assert model.statements.ode_system.find_depot(model.statements).name == 'DEPOT'
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    assert model.statements.ode_system.find_depot(model.statements) is None


def test_find_depot_multi_outflows(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    odes = model.statements.ode_system
    cb = CompartmentalSystemBuilder(odes)
    comp = Compartment.create(name='A')
    cb.add_compartment(comp)
    cb.add_flow(comp, output, 'X')
    central = odes.central_compartment
    cb.add_flow(central, comp, 'Y')
    cs = CompartmentalSystem(cb)

    assert cs.find_depot(model.statements) is None


def test_find_peripheral_compartments(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    odes = model.statements.ode_system
    assert odes.find_peripheral_compartments() == []

    central = odes.find_compartment('CENTRAL')
    peripheral = Compartment.create('PERIPHERAL1')

    cb = CompartmentalSystemBuilder(odes)
    cb.add_compartment(peripheral)
    cb.add_flow(central, peripheral, S('X'))
    cb.add_flow(peripheral, central, S('X'))

    odes = CompartmentalSystem(cb)

    peripherals = odes.find_peripheral_compartments()
    assert len(peripherals) == 1
    assert peripherals[0].name == 'PERIPHERAL1'

    with pytest.raises(ValueError):
        model.statements.ode_system.find_peripheral_compartments('x')


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


def test_get_compartment_inflows(load_model_for_test, pheno_path, testdata):
    model = load_model_for_test(pheno_path)
    assert len(model.statements.ode_system.get_compartment_inflows(output)) == 1

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    assert len(model.statements.ode_system.get_compartment_inflows('CENTRAL')) == 1

    with pytest.raises(ValueError):
        model.statements.ode_system.get_compartment_inflows('NOTINMODEL')


def test_cs_replace(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    odes = model.statements.ode_system

    central = odes.find_compartment('CENTRAL')
    peripheral = Compartment.create('PERIPHERAL1')

    cb = CompartmentalSystemBuilder(odes)
    cb.add_compartment(peripheral)
    cb.add_flow(central, peripheral, S('X'))
    cb.add_flow(peripheral, central, S('X'))

    assert len(odes.compartment_names) == 1
    odes = odes.replace(builder=cb)
    assert len(odes.compartment_names) == 2

    assert odes.t == Expr.symbol('t')
    odes = odes.replace(t=Expr.symbol('b'))
    assert odes.t == Expr.symbol('b')


def test_before_odes(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    before_ode = model.statements.before_odes
    assert before_ode[-1].symbol.name == 'S1'


def test_full_expression(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    expr = model.statements.before_odes.full_expression("CL")
    assert expr == Expr.symbol("PTVCL") * Expr.symbol("WGT") * (Expr.symbol("ETA_1").exp())
    with pytest.raises(ValueError):
        model.statements.full_expression("Y")


def test_to_explicit_ode_system(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    cs = model.statements.ode_system
    assert len(cs.eqs) == 1

    assert cs.amounts == Matrix([Expr.function('A_CENTRAL', 't')])


def test_repr(load_model_for_test, testdata):
    def _remove_dose_lines(repr_str):
        no_dose = [line for line in repr_str.split('\n') if not line.startswith('Bolus')]
        return '\n'.join(no_dose)

    cb = CompartmentalSystemBuilder()
    dose = Bolus(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL', doses=(dose,))
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    cs = CompartmentalSystem(cb)
    cs_repr = repr(cs)
    assert 'Bolus(AMT, admid=1) → CENTRAL' in cs_repr
    assert all(comp in _remove_dose_lines(cs_repr) for comp in cs.compartment_names)

    cb = CompartmentalSystemBuilder(cs)
    depot = Compartment.create('DEPOT')
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cb.move_dose(central, depot)  # Starts with dose
    cs_depot = CompartmentalSystem(cb)
    cs_depot_repr = repr(cs_depot)
    assert 'Bolus(AMT, admid=1) → DEPOT' in cs_depot_repr
    assert all(comp in _remove_dose_lines(cs_depot_repr) for comp in cs_depot.compartment_names)

    cb = CompartmentalSystemBuilder(cs)
    peripheral1 = Compartment.create('PERIPHERAL1')
    cb.add_compartment(peripheral1)
    cb.add_flow(peripheral1, central, S('X1'))  # Needs to be bidirectional
    cb.add_flow(central, peripheral1, S('Y1'))
    cs_p1 = CompartmentalSystem(cb)
    assert all(comp in _remove_dose_lines(repr(cs_p1)) for comp in cs_p1.compartment_names)

    cb = CompartmentalSystemBuilder(cs_p1)
    peripheral2 = Compartment.create('PERIPHERAL2')
    cb.add_compartment(peripheral2)
    cb.add_flow(peripheral2, central, S('X2'))
    cb.add_flow(central, peripheral2, S('Y2'))
    cs_p2 = CompartmentalSystem(cb)
    assert all(comp in _remove_dose_lines(repr(cs_p2)) for comp in cs_p2.compartment_names)

    cb = CompartmentalSystemBuilder(cs_depot)
    central = cs_depot.central_compartment
    dose = Bolus(Expr.symbol('AMT'), admid=2)
    cb.set_dose(central, dose)
    cs_multi_dose = CompartmentalSystem(cb)
    cs_multi_dose_repr = repr(cs_multi_dose)
    assert 'Bolus(AMT, admid=1) → DEPOT' in cs_multi_dose_repr
    assert 'Bolus(AMT, admid=2) → CENTRAL' in cs_multi_dose_repr
    assert all(
        comp in _remove_dose_lines(cs_multi_dose_repr) for comp in cs_multi_dose.compartment_names
    )

    cb = CompartmentalSystemBuilder(cs)
    central = cs.central_compartment
    comp = Compartment.create('A')
    cb.add_compartment(comp)
    cb.add_flow(comp, output, S('X'))
    cb.add_flow(central, comp, S('Y'))
    cs_extra_comp = CompartmentalSystem(cb)
    cs_extra_comp_repr = repr(cs_extra_comp)
    assert 'Bolus(AMT, admid=1) → CENTRAL' in cs_extra_comp_repr
    assert all(
        comp in _remove_dose_lines(cs_extra_comp_repr) for comp in cs_extra_comp.compartment_names
    )


def test_repr_latex():
    s1 = Assignment.create(S('KA'), S('X') + S('Y'))
    latex = s1._repr_latex_()
    assert latex == r'$KA = X + Y$'


def test_repr_html():
    s1 = Assignment.create(S('KA'), S('X') + S('Y'))
    stats = Statements([s1])
    html = stats._repr_html_()
    assert 'X + Y' in html

    cb = CompartmentalSystemBuilder()
    dose = Bolus.create('AMT')
    central = Compartment.create('CENTRAL', doses=(dose,))
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
    dose = Bolus(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL', doses=(dose,))
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    depot = Compartment.create('DEPOT')
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cm = CompartmentalSystem(cb)
    assert not cm.find_compartment('DEPOT').doses
    assert cm.central_compartment.doses[0] == dose
    cb.move_dose(central, depot)
    cm2 = CompartmentalSystem(cb)
    assert cm2.find_compartment('DEPOT').doses[0] == dose
    assert not cm2.central_compartment.doses
    assert hash(cm) != hash(cm2)


def test_add_dose():
    cb = CompartmentalSystemBuilder()
    dose1 = Bolus.create(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    cb.add_dose(central, dose1)
    cs = CompartmentalSystem(cb)
    assert cs.central_compartment.doses == (dose1,)

    dose2 = Infusion.create('AMT', rate='R1')
    cb.add_dose(cs.central_compartment, dose2)
    cs = CompartmentalSystem(cb)
    assert cs.central_compartment.doses == (
        dose2,
        dose1,
    )

    with pytest.raises(ValueError):
        cb.add_dose(None, dose1)
    with pytest.raises(ValueError):
        cb.add_dose(cs.central_compartment, None)


def test_set_dose():
    cb = CompartmentalSystemBuilder()
    dose1 = Bolus.create(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    cb.set_dose(central, dose1)
    cs = CompartmentalSystem(cb)
    assert cs.central_compartment.doses == (dose1,)

    dose2 = Infusion.create('AMT', rate='R1')
    cb.set_dose(cs.central_compartment, dose2)
    cs = CompartmentalSystem(cb)
    assert cs.central_compartment.doses == (dose2,)

    with pytest.raises(ValueError):
        cb.set_dose(None, dose1)


def test_remove_dose():
    cb = CompartmentalSystemBuilder()
    dose1 = Bolus.create(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    depot = Compartment.create('DEPOT', doses=(dose1,))
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cs = CompartmentalSystem(cb)
    assert cs.find_compartment('DEPOT').doses == (dose1,)

    cb.remove_dose(depot)
    cs = CompartmentalSystem(cb)
    assert cs.find_compartment('DEPOT').doses == tuple()

    dose2 = Infusion.create('AMT', rate='R1', admid=2)
    cb.add_dose(cs.find_compartment('DEPOT'), dose1)
    cb.add_dose(cs.central_compartment, dose2)
    cs = CompartmentalSystem(cb)
    assert cs.find_compartment('DEPOT').doses == (dose1,)
    assert cs.central_compartment.doses == (dose2,)

    cb1 = CompartmentalSystemBuilder(cs)
    cb1.remove_dose(cs.central_compartment)
    cs1 = CompartmentalSystem(cb1)
    assert cs1.find_compartment('DEPOT').doses == (dose1,)
    assert cs1.central_compartment.doses == tuple()

    cb2 = CompartmentalSystemBuilder(cs)
    cb2.remove_dose(cs.central_compartment, admid=2)
    cs2 = CompartmentalSystem(cb2)
    assert cs2.find_compartment('DEPOT').doses == (dose1,)
    assert cs2.central_compartment.doses == tuple()

    with pytest.raises(ValueError):
        cb.remove_dose(None, dose1)


def test_move_dose():
    cb = CompartmentalSystemBuilder()
    dose1 = Bolus.create(Expr.symbol('AMT'))
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('K'))
    depot = Compartment.create('DEPOT', doses=(dose1,))
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('KA'))
    cs = CompartmentalSystem(cb)
    assert cs.find_compartment('DEPOT').doses == (dose1,)
    assert cs.central_compartment.doses == tuple()

    cb1 = CompartmentalSystemBuilder(cs)
    cb1.move_dose(depot, central)
    cs1 = CompartmentalSystem(cb1)
    assert cs1.find_compartment('DEPOT').doses == tuple()
    assert cs1.central_compartment.doses == (dose1,)

    cb2 = CompartmentalSystemBuilder(cs)
    dose2 = Infusion.create('AMT', rate='R1', admid=2)
    central = cb2.add_dose(cs.central_compartment, dose2)
    depot, central = cb2.move_dose(depot, central)
    cs2 = CompartmentalSystem(cb2)
    assert cs2.find_compartment('DEPOT').doses == tuple()
    assert cs2.central_compartment.doses == (dose2, dose1)

    central, _ = cb2.move_dose(central, depot, admid=2)
    cs3 = CompartmentalSystem(cb2)
    assert cs3.find_compartment('DEPOT').doses == (dose2,)
    assert cs3.central_compartment.doses == (dose1,)

    central = cb2.remove_dose(central)
    with pytest.raises(ValueError):
        cb.move_dose(None, central)
    with pytest.raises(ValueError):
        cb.move_dose(central, None)
    with pytest.raises(ValueError):
        cb2.move_dose(central, depot)


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
    assert repr(comp) == "Compartment(CENTRAL, amount=A_CENTRAL(t), lag_time=LT)"


def test_compartment_names(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert model.statements.ode_system.compartment_names == ['CENTRAL']

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    assert model.statements.ode_system.compartment_names == ['DEPOT', 'CENTRAL']

    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('X'))
    dose = Bolus.create('AMT')
    cb.set_dose(central, dose=(dose,))
    effect1 = Compartment.create('EFFECT1', input=0)
    cb.add_compartment(effect1)
    cb.add_flow(effect1, output, S('Y'))
    effect2 = Compartment.create('EFFECT2', input=S('Y') * central.amount)
    cb.add_compartment(effect2)
    cb.add_flow(effect2, output, S('Y'))
    cs = CompartmentalSystem(cb)
    assert cs.compartment_names == ['CENTRAL', 'EFFECT2', 'EFFECT1']

    # Compartment upstream of dosing
    cb = CompartmentalSystemBuilder()
    dose = Bolus.create('AMT')
    central = Compartment.create('CENTRAL', doses=(dose,))
    cb.add_compartment(central)
    cb.add_flow(central, output, S('X'))
    comp = Compartment.create('COMP')
    cb.add_compartment(comp)
    cb.add_flow(comp, central, S('Y'))
    cs = CompartmentalSystem(cb)
    assert cs.compartment_names == ['CENTRAL', 'COMP']


def test_assignment_create_numeric(load_model_for_test, testdata):
    with pytest.raises(AttributeError):
        Assignment('X', 1).free_symbols
    assert Assignment.create('X', 1).free_symbols
    with pytest.raises(AttributeError):
        Assignment('X', 1.0).free_symbols
    assert Assignment.create('X', 1.0).free_symbols


def test_multi_dosing_compartment(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_first_order_absorption(model)

    ode = model.statements.ode_system
    cb = CompartmentalSystemBuilder(ode)
    cb.set_dose(cb.find_compartment("CENTRAL"), cb.find_compartment("DEPOT").doses)
    ode = CompartmentalSystem(cb)
    assert ode.dosing_compartments[0].name == "DEPOT"

    dose = Bolus.create('AMT')
    comp = Compartment.create('A', doses=(dose,))
    central = ode.find_compartment('CENTRAL')
    cb.add_compartment(comp)
    cb.add_flow(comp, central, S('X'))
    cb.add_flow(central, comp, S('Y'))
    ode = CompartmentalSystem(cb)
    assert ode.dosing_compartments[-1].name == "CENTRAL"


def test_get_n_connected(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')

    odes = model.statements.ode_system
    central = odes.find_compartment('CENTRAL')
    peripheral = Compartment.create('PERIPHERAL1')

    cb = CompartmentalSystemBuilder(odes)
    cb.add_compartment(peripheral)
    cb.add_flow(central, peripheral, S('X'))
    cb.add_flow(peripheral, central, S('X'))

    odes = CompartmentalSystem(cb)
    assert odes.get_n_connected(peripheral) == 1
    assert odes.get_n_connected(central) == 2


def test_output():
    output_new = Output()
    assert output == output_new
    assert output != 'x'


def test_to_compartmental_system(load_model_for_test, testdata):
    def _create_names(cs):
        return {Expr.function(f'A_{name}', cs.t): name for name in cs.compartment_names}

    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, S('X'))
    depot = Compartment.create('DEPOT')
    cb.add_compartment(depot)
    cb.add_flow(depot, central, S('Y'))
    cs_from_cb = CompartmentalSystem(cb)
    cs_from_func = to_compartmental_system(_create_names(cs_from_cb), cs_from_cb.eqs)
    assert cs_from_cb.compartment_names == cs_from_func.compartment_names
    assert cs_from_cb.get_flow(central, output) == cs_from_func.get_flow(central, output)
    assert cs_from_cb.get_flow(depot, central) == cs_from_func.get_flow(depot, central)

    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.add_flow(central, output, 0)
    cs_from_cb = CompartmentalSystem(cb)
    cs_from_func = to_compartmental_system(_create_names(cs_from_cb), cs_from_cb.eqs)
    assert cs_from_cb.compartment_names == cs_from_func.compartment_names
    assert cs_from_cb.get_flow(central, output) == cs_from_func.get_flow(central, output) == 0

    # Second order absorption
    k, v = S('k'), S('V')
    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL')
    cb.add_flow(central, output, k)
    complex = Compartment.create('COMPLEX')
    target = Compartment.create('TARGET')
    cb.add_flow(target, complex, central.amount / v)
    cb.add_flow(complex, target, k)
    cb.add_flow(target, output, k)
    cb.add_flow(complex, output, k)
    cb.set_input(target, v)
    cb.set_input(central, k * complex.amount - central.amount * target.amount / v)
    cs_from_cb = CompartmentalSystem(cb)
    cs_from_func = to_compartmental_system(_create_names(cs_from_cb), cs_from_cb.eqs)
    assert cs_from_cb.compartment_names == cs_from_func.compartment_names
    # assert cs_from_cb.zero_order_inputs == cs_from_func.zero_order_inputs

    # Second order input
    central = cs_from_cb.find_compartment('CENTRAL')
    complex = cs_from_cb.find_compartment('COMPLEX')
    cb.set_input(complex, central.amount / target.amount)
    cs_from_cb = CompartmentalSystem(cb)
    cs_from_func = to_compartmental_system(_create_names(cs_from_cb), cs_from_cb.eqs)
    assert cs_from_cb.compartment_names == cs_from_func.compartment_names

    # No output
    cb = CompartmentalSystemBuilder()
    central = Compartment.create('CENTRAL')
    cb.add_compartment(central)
    cb.set_input(central, S('X'))
    cs_from_cb = CompartmentalSystem(cb)
    cs_from_func = to_compartmental_system(_create_names(cs_from_cb), cs_from_cb.eqs)
    assert cs_from_cb.compartment_names == cs_from_func.compartment_names
