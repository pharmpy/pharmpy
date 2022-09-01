import pytest
import sympy

from pharmpy import (
    Assignment,
    Model,
    Parameter,
    Parameters,
    RandomVariable,
    RandomVariables,
    Statements,
)
from pharmpy.modeling import (
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    cleanup_model,
    get_individual_parameters,
    get_individual_prediction_expression,
    get_observation_expression,
    get_pk_parameters,
    get_population_prediction_expression,
    greekify_model,
    has_random_effect,
    make_declarative,
    mu_reference_model,
    read_model,
    read_model_from_string,
    simplify_expression,
    solve_ode_system,
)


def s(x):
    return sympy.Symbol(x)


def test_get_observation_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_observation_expression(model)
    assert expr == s('D_EPSETA1_2') * s('EPS(1)') * (s('ETA(2)') - s('OETA2')) + s('D_ETA1') * (
        s('ETA(1)') - s('OETA1')
    ) + s('D_ETA2') * (s('ETA(2)') - s('OETA2')) + s('EPS(1)') * (
        s('D_EPS1') + s('D_EPSETA1_1') * (s('ETA(1)') - s('OETA1'))
    ) + s(
        'OPRED'
    )


def test_get_individual_prediction_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_individual_prediction_expression(model)
    assert expr == s('D_ETA1') * (s('ETA(1)') - s('OETA1')) + s('D_ETA2') * (
        s('ETA(2)') - s('OETA2')
    ) + s('OPRED')


def test_get_population_prediction_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_population_prediction_expression(model)
    assert expr == -s('D_ETA1') * s('OETA1') - s('D_ETA2') * s('OETA2') + s('OPRED')


def test_calculate_eta_gradient_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_eta_gradient_expression(model)
    assert expr == [s('D_ETA1'), s('D_ETA2')]


def test_calculate_epsilon_gradient_expression(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_epsilon_gradient_expression(model)
    assert expr == [
        s('D_EPS1')
        + s('D_EPSETA1_1') * (s('ETA(1)') - s('OETA1'))
        + s('D_EPSETA1_2') * (s('ETA(2)') - s('OETA2'))
    ]


def test_mu_reference_model_full(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$OMEGA DIAGONAL(2)
0.0309626 ; IVCL
0.031128 ; IVV
$SIGMA 0.013241 ;sigma
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    mu_reference_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
MU_1 = LOG(TVCL)
CL = EXP(ETA(1) + MU_1)
MU_2 = LOG(TVV)
V = EXP(ETA(2) + MU_2)
S1=V
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$OMEGA DIAGONAL(2)
0.0309626 ; IVCL
0.031128 ; IVV
$SIGMA 0.013241 ;sigma
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


@pytest.mark.usefixtures('testdata')
@pytest.mark.parametrize(
    'statements,correct',
    [
        (
            [Assignment.create('CL', s('THETA(1)') + s('ETA(1)'))],
            [
                Assignment.create('mu_1', s('THETA(1)')),
                Assignment.create('CL', s('mu_1') + s('ETA(1)')),
            ],
        ),
        (
            [
                Assignment.create(
                    'CL', s('THETA(1)') * s('AGE') ** s('THETA(2)') * sympy.exp(s('ETA(1)'))
                ),
                Assignment.create('V', s('THETA(3)') * sympy.exp(s('ETA(2)'))),
            ],
            [
                Assignment.create('mu_1', sympy.log(s('THETA(1)') * s('AGE') ** s('THETA(2)'))),
                Assignment.create('CL', sympy.exp(s('mu_1') + s('ETA(1)'))),
                Assignment.create('mu_2', sympy.log(s('THETA(3)'))),
                Assignment.create('V', sympy.exp(s('mu_2') + s('ETA(2)'))),
            ],
        ),
        (
            [Assignment.create('CL', s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))],
            [Assignment.create('CL', s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))],
        ),
    ],
)
def test_mu_reference_model_generic(testdata, statements, correct):
    model = Model()
    model.statements = Statements(statements)
    eta1 = RandomVariable.normal('ETA(1)', 'iiv', 0, s('omega1'))
    eta2 = RandomVariable.normal('ETA(2)', 'iiv', 0, s('omega2'))
    rvs = RandomVariables([eta1, eta2])
    model.random_variables = rvs
    th1 = Parameter('THETA(1)', 2, lower=1)
    th2 = Parameter('THETA(2)', 2, lower=1)
    th3 = Parameter('THETA(3)', 2, lower=1)
    params = Parameters([th1, th2, th3])
    model.parameters = params
    mu_reference_model(model)
    assert model.statements == Statements(correct)


def test_simplify_expression():
    model = Model()
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9, fix=True)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, x * y) == 9.0 * x

    p1 = Parameter('x', 3, lower=0.001)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, abs(x)) == x

    p1 = Parameter('x', 3, lower=0)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, sympy.Piecewise((2, sympy.Ge(x, 0)), (56, True))) == 2

    p1 = Parameter('x', -3, upper=-1)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, abs(x)) == -x

    p1 = Parameter('x', -3, upper=0)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, sympy.Piecewise((2, sympy.Le(x, 0)), (56, True))) == 2

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9)
    pset = Parameters([p1, p2])
    model.parameters = pset
    assert simplify_expression(model, x * y) == x * y


def test_solve_ode_system(pheno):
    model = pheno.copy()
    solve_ode_system(model)
    assert sympy.Symbol('t') in model.statements[8].free_symbols


def test_make_declarative(pheno):
    model = pheno.copy()
    make_declarative(model)
    assert model.statements[3].expression == sympy.Piecewise(
        (s('WGT') * s('THETA(2)') * (s('THETA(3)') + 1), sympy.Lt(s('APGR'), 5)),
        (s('WGT') * s('THETA(2)'), True),
    )


def test_cleanup_model(pheno):
    model = pheno.copy()
    cleanup_model(model)
    assert model.statements.after_odes[1].symbol != s('W')


def test_greekify_model(pheno):
    model = pheno.copy()
    cleanup_model(model)
    greekify_model(model)
    assert s('theta_1') in model.statements[2].free_symbols


@pytest.mark.parametrize(
    ('model_path', 'kind', 'expected'),
    (
        ('nonmem/pheno.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'all', ['CL', 'V']),
        ('nonmem/models/mox1.mod', 'all', ['CL', 'KA', 'V']),
        ('nonmem/models/mox2.mod', 'all', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox_2comp.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pef.mod', 'all', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'all', ['CL', 'V']),
        ('nonmem/models/pheno_trans1.mod', 'all', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'all', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'all', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno.mod', 'absorption', []),
        ('nonmem/pheno_real.mod', 'absorption', []),
        ('nonmem/models/mox1.mod', 'absorption', ['KA']),
        ('nonmem/models/mox2.mod', 'absorption', ['MAT']),
        ('nonmem/models/mox_2comp.mod', 'absorption', []),
        ('nonmem/models/pef.mod', 'absorption', []),
        ('nonmem/models/pheno_advan3_trans1.mod', 'absorption', []),
        ('nonmem/models/pheno_noifs.mod', 'absorption', []),
        ('nonmem/models/pheno_trans1.mod', 'absorption', []),
        ('nonmem/secondary_parameters/run1.mod', 'absorption', []),
        ('nonmem/secondary_parameters/run2.mod', 'absorption', []),
        ('nonmem/qa/iov.mod', 'absorption', []),
        ('nonmem/pheno.mod', 'distribution', ['V']),
        ('nonmem/pheno_real.mod', 'distribution', ['V']),
        ('nonmem/models/mox1.mod', 'distribution', ['V']),
        ('nonmem/models/mox2.mod', 'distribution', ['VC']),
        ('nonmem/models/mox_2comp.mod', 'distribution', ['K12', 'K21']),
        ('nonmem/models/pef.mod', 'distribution', []),
        ('nonmem/models/pheno_advan3_trans1.mod', 'distribution', ['K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'distribution', ['V']),
        ('nonmem/models/pheno_trans1.mod', 'distribution', []),
        ('nonmem/secondary_parameters/run1.mod', 'distribution', ['V']),
        ('nonmem/secondary_parameters/run2.mod', 'distribution', ['V']),
        ('nonmem/qa/iov.mod', 'distribution', ['V']),
        ('nonmem/pheno.mod', 'elimination', ['CL']),
        ('nonmem/pheno_real.mod', 'elimination', ['CL']),
        ('nonmem/models/mox1.mod', 'elimination', ['CL']),
        ('nonmem/models/mox2.mod', 'elimination', ['CL']),
        ('nonmem/models/mox_2comp.mod', 'elimination', ['K']),
        ('nonmem/models/pef.mod', 'elimination', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'elimination', ['K']),
        ('nonmem/models/pheno_noifs.mod', 'elimination', ['CL']),
        ('nonmem/models/pheno_trans1.mod', 'elimination', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'elimination', ['CL']),
        ('nonmem/secondary_parameters/run2.mod', 'elimination', ['CL']),
        ('nonmem/qa/iov.mod', 'elimination', ['CL']),
        ('nonmem/modeling/pheno_1transit.mod', 'all', ['CL', 'V', 'K12', 'K23', 'K34', 'K43']),
        ('nonmem/modeling/pheno_1transit.mod', 'absorption', ['K12', 'K23']),
        ('nonmem/modeling/pheno_1transit.mod', 'distribution', ['V', 'K34', 'K43']),
        ('nonmem/modeling/pheno_1transit.mod', 'elimination', ['CL']),
        (
            'nonmem/modeling/pheno_2transits.mod',
            'all',
            ['CL', 'V', 'K12', 'K23', 'K34', 'K45', 'K54'],
        ),
        ('nonmem/modeling/pheno_2transits.mod', 'absorption', ['K12', 'K23', 'K34']),
        ('nonmem/modeling/pheno_2transits.mod', 'distribution', ['V', 'K45', 'K54']),
        ('nonmem/modeling/pheno_2transits.mod', 'elimination', ['CL']),
        ('nonmem/modeling/pheno_advan5_nodepot.mod', 'all', ['CL', 'V', 'K12', 'K21']),
        ('nonmem/modeling/pheno_advan5_nodepot.mod', 'absorption', []),
        ('nonmem/modeling/pheno_advan5_nodepot.mod', 'distribution', ['V', 'K12', 'K21']),
        ('nonmem/modeling/pheno_advan5_nodepot.mod', 'elimination', ['CL']),
        ('nonmem/modeling/pheno_advan5_depot.mod', 'all', ['CL', 'V', 'K12', 'K23', 'K32']),
        ('nonmem/modeling/pheno_advan5_depot.mod', 'absorption', ['K12']),
        ('nonmem/modeling/pheno_advan5_depot.mod', 'distribution', ['V', 'K23', 'K32']),
        ('nonmem/modeling/pheno_advan5_depot.mod', 'elimination', ['CL']),
        (
            'nonmem/modeling/transit_indirect_reabsorption.mod',
            'all',
            ['CL', 'V', 'K12', 'K23', 'KA', 'K45', 'K56', 'K64'],
        ),
        ('nonmem/modeling/transit_indirect_reabsorption.mod', 'absorption', ['K12', 'K23', 'KA']),
        (
            'nonmem/modeling/transit_indirect_reabsorption.mod',
            'distribution',
            ['V', 'K45', 'K56', 'K64'],
        ),
        ('nonmem/modeling/transit_indirect_reabsorption.mod', 'elimination', ['CL']),
    ),
    ids=repr,
)
def test_get_pk_parameters(testdata, model_path, kind, expected):
    model = read_model(testdata / model_path)
    assert set(get_pk_parameters(model, kind)) == set(expected)


@pytest.mark.parametrize(
    ('model_path', 'level', 'expected'),
    (
        ('nonmem/pheno.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'all', ['CL', 'V']),
        ('nonmem/models/mox1.mod', 'all', ['CL', 'KA', 'V']),
        ('nonmem/models/mox2.mod', 'all', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox_2comp.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pef.mod', 'all', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'all', ['CL', 'V']),
        ('nonmem/models/pheno_trans1.mod', 'all', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'all', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'all', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'iiv', ['CL', 'V']),
        ('nonmem/models/mox1.mod', 'iiv', ['CL', 'KA', 'V']),
        ('nonmem/models/mox2.mod', 'iiv', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox_2comp.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/pef.mod', 'iiv', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'iiv', ['CL', 'V']),
        ('nonmem/models/pheno_trans1.mod', 'iiv', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'iiv', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'iiv', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iov', []),
        ('nonmem/pheno_real.mod', 'iov', []),
        ('nonmem/models/mox1.mod', 'iov', []),
        ('nonmem/models/mox2.mod', 'iov', []),
        ('nonmem/models/mox_2comp.mod', 'iov', []),
        ('nonmem/models/pef.mod', 'iov', []),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iov', []),
        ('nonmem/models/pheno_noifs.mod', 'iov', []),
        ('nonmem/models/pheno_trans1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run2.mod', 'iov', []),
        ('nonmem/qa/iov.mod', 'iov', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'all', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iov', []),
        ('nonmem/modeling/pheno_2transits.mod', 'all', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iov', []),
    ),
    ids=repr,
)
def test_get_individual_parameters(testdata, model_path, level, expected):
    model = read_model(testdata / model_path)
    assert set(get_individual_parameters(model, level)) == set(expected)


@pytest.mark.parametrize(
    ('model_path', 'level', 'expected'),
    (
        ('nonmem/pheno.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'all', ['CL', 'V']),
        ('nonmem/models/mox1.mod', 'all', ['CL', 'KA', 'V']),
        ('nonmem/models/mox2.mod', 'all', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox_2comp.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pef.mod', 'all', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'all', ['CL', 'V']),
        ('nonmem/models/pheno_trans1.mod', 'all', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'all', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'all', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'iiv', ['CL', 'V']),
        ('nonmem/models/mox1.mod', 'iiv', ['CL', 'KA', 'V']),
        ('nonmem/models/mox2.mod', 'iiv', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox_2comp.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/pef.mod', 'iiv', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_noifs.mod', 'iiv', ['CL', 'V']),
        ('nonmem/models/pheno_trans1.mod', 'iiv', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'iiv', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'iiv', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iov', []),
        ('nonmem/pheno_real.mod', 'iov', []),
        ('nonmem/models/mox1.mod', 'iov', []),
        ('nonmem/models/mox2.mod', 'iov', []),
        ('nonmem/models/mox_2comp.mod', 'iov', []),
        ('nonmem/models/pef.mod', 'iov', []),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iov', []),
        ('nonmem/models/pheno_noifs.mod', 'iov', []),
        ('nonmem/models/pheno_trans1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run2.mod', 'iov', []),
        ('nonmem/qa/iov.mod', 'iov', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'all', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iov', []),
        ('nonmem/modeling/pheno_2transits.mod', 'all', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iov', []),
    ),
    ids=repr,
)
def test_has_random_effect(testdata, model_path, level, expected):
    model = read_model(testdata / model_path)
    params_with_random_effect = set(expected)
    for param in get_pk_parameters(model):
        if param in params_with_random_effect:
            assert has_random_effect(model, param, level)
        else:
            assert not has_random_effect(model, param, level)
