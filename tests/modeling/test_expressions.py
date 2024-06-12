from functools import partial

import pytest
import sympy

from pharmpy.basic import Expr
from pharmpy.model import (
    Assignment,
    DataInfo,
    Model,
    NormalDistribution,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
)
from pharmpy.modeling import (
    add_covariate_effect,
    add_effect_compartment,
    add_indirect_effect,
    add_metabolite,
    add_peripheral_compartment,
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    cleanup_model,
    create_basic_pk_model,
    display_odes,
    get_dv_symbol,
    get_individual_parameters,
    get_individual_prediction_expression,
    get_observation_expression,
    get_parameter_rv,
    get_pd_parameters,
    get_pk_parameters,
    get_population_prediction_expression,
    get_rv_parameters,
    greekify_model,
    has_mu_reference,
    has_random_effect,
    is_linearized,
    is_real,
    make_declarative,
    mu_reference_model,
    read_model_from_string,
    remove_covariate_effect,
    set_direct_effect,
    set_first_order_absorption,
    set_transit_compartments,
    simplify_expression,
    solve_ode_system,
)


def s(x):
    return Expr.symbol(x)


def test_get_observation_expression(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_observation_expression(model)
    assert expr == s('D_EPSETA1_2') * s('EPS_1') * (s('ETA_2') - s('OETA2')) + s('D_ETA1') * (
        s('ETA_1') - s('OETA1')
    ) + s('D_ETA2') * (s('ETA_2') - s('OETA2')) + s('EPS_1') * (
        s('D_EPS1') + s('D_EPSETA1_1') * (s('ETA_1') - s('OETA1'))
    ) + s(
        'OPRED'
    )


def test_get_individual_prediction_expression(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_individual_prediction_expression(model)
    assert expr == s('D_ETA1') * (s('ETA_1') - s('OETA1')) + s('D_ETA2') * (
        s('ETA_2') - s('OETA2')
    ) + s('OPRED')


def test_get_population_prediction_expression(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = get_population_prediction_expression(model)
    assert expr == -s('D_ETA1') * s('OETA1') - s('D_ETA2') * s('OETA2') + s('OPRED')


def test_calculate_eta_gradient_expression(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_eta_gradient_expression(model)
    assert expr == [s('D_ETA1'), s('D_ETA2')]


def test_calculate_epsilon_gradient_expression(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real_linbase.mod')
    expr = calculate_epsilon_gradient_expression(model)
    assert expr == [
        s('D_EPS1')
        + s('D_EPSETA1_1') * (s('ETA_1') - s('OETA1'))
        + s('D_EPSETA1_2') * (s('ETA_2') - s('OETA2'))
    ]


def test_mu_reference_model_full():
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
    model = mu_reference_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
MU_1 = LOG(TVCL)
CL = EXP(MU_1 + ETA(1))
MU_2 = LOG(TVV)
V = EXP(MU_2 + ETA(2))
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
    assert model.code == correct


@pytest.mark.usefixtures('testdata')
@pytest.mark.parametrize(
    'statements,correct',
    [
        (
            [Assignment(s('CL'), s('THETA(1)') + s('ETA(1)'))],
            [
                Assignment(s('mu_1'), s('THETA(1)')),
                Assignment(s('CL'), s('mu_1') + s('ETA(1)')),
            ],
        ),
        (
            [
                Assignment(s('CL'), s('THETA(1)') * s('AGE') ** s('THETA(2)') * s('ETA(1)').exp()),
                Assignment(s('V'), s('THETA(3)') * s('ETA(2)').exp()),
            ],
            [
                Assignment(s('mu_1'), (s('THETA(1)') * s('AGE') ** s('THETA(2)')).log()),
                Assignment(s('CL'), (s('mu_1') + s('ETA(1)')).exp()),
                Assignment(s('mu_2'), s('THETA(3)').log()),
                Assignment(s('V'), (s('mu_2') + s('ETA(2)')).exp()),
            ],
        ),
        (
            [Assignment(s('CL'), s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))],
            [Assignment(s('CL'), s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))],
        ),
    ],
)
def test_mu_reference_model_generic(statements, correct):
    model = Model()
    datainfo = DataInfo.create(['AGE'])
    eta1 = NormalDistribution.create('ETA(1)', 'iiv', 0, s('omega1'))
    eta2 = NormalDistribution.create('ETA(2)', 'iiv', 0, s('omega2'))
    rvs = RandomVariables.create([eta1, eta2])
    th1 = Parameter('THETA(1)', 2, lower=1)
    th2 = Parameter('THETA(2)', 2, lower=1)
    th3 = Parameter('THETA(3)', 2, lower=1)
    params = Parameters((th1, th2, th3))
    model = model.replace(
        statements=Statements(statements),
        parameters=params,
        random_variables=rvs,
        datainfo=datainfo,
    )
    model = mu_reference_model(model)
    assert model.statements == Statements(correct)


def test_mu_reference_covariate_effect(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = remove_covariate_effect(model, "CL", "WGT")  # Define using template instead
    model = add_covariate_effect(model, "CL", "WGT", "pow")

    model = mu_reference_model(model)

    assert (
        model.code
        == """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
WGT_MEDIAN = 1.30000000000000
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL = THETA(1)
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL = TVCL
CLWGT = (WGT/WGT_MEDIAN)**THETA(4)
MU_1 = LOG(CL*CLWGT)
CL = EXP(MU_1 + ETA(1))
MU_2 = LOG(TVV)
V = EXP(MU_2 + ETA(2))
S1=V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$THETA (-.99,.1)
$THETA  (-100,0.001,100000) ; POP_CLWGT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )


def test_add_covariate_effect_on_mu_referenced_model(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = remove_covariate_effect(model, "CL", "WGT")  # Define using template instead
    model = mu_reference_model(model)

    model = add_covariate_effect(model, "CL", "WGT", "pow")
    assert (
        model.code
        == """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
WGT_MEDIAN = 1.30000000000000
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL = THETA(1)
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CLWGT = (WGT/WGT_MEDIAN)**THETA(4)
MU_1 = LOG(CLWGT*TVCL)
CL = EXP(MU_1 + ETA(1))
MU_2 = LOG(TVV)
V = EXP(MU_2 + ETA(2))
S1=V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$THETA (-.99,.1)
$THETA  (-100,0.001,100000) ; POP_CLWGT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )

    model = remove_covariate_effect(model, "CL", "WGT")
    assert (
        model.code
        == """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL = THETA(1)
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
MU_1 = LOG(TVCL)
CL = EXP(MU_1 + ETA(1))
MU_2 = LOG(TVV)
V = EXP(MU_2 + ETA(2))
S1=V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$THETA (-.99,.1)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )


def test_has_mu_reference(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    assert not has_mu_reference(model)
    model = mu_reference_model(model)
    assert has_mu_reference(model)


def test_simplify_expression():
    model = Model()
    x = Expr.symbol('x')
    y = Expr.symbol('y')

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9, fix=True)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, x * y) == 9.0 * x

    p1 = Parameter('x', 3, lower=0.001)
    p2 = Parameter('y', 9)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, abs(x)) == x

    p1 = Parameter('x', 3, lower=0)
    p2 = Parameter('y', 9)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, Expr.piecewise((2, sympy.Ge(x, 0)), (56, True))) == 2

    p1 = Parameter('x', -3, upper=-1)
    p2 = Parameter('y', 9)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, abs(x)) == -x

    p1 = Parameter('x', -3, upper=0)
    p2 = Parameter('y', 9)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, Expr.piecewise((2, sympy.Le(x, 0)), (56, True))) == 2

    p1 = Parameter('x', 3)
    p2 = Parameter('y', 9)
    pset = Parameters((p1, p2))
    model = model.replace(parameters=pset)
    assert simplify_expression(model, x * y) == x * y


def test_solve_ode_system(pheno):
    model = solve_ode_system(pheno)
    assert Expr.symbol('t') in model.statements[8].free_symbols


def test_make_declarative(pheno):
    model = make_declarative(pheno)
    assert model.statements[3].expression == Expr.piecewise(
        (s('WGT') * s('PTVV') * (s('THETA_3') + 1), sympy.Lt(s('APGR'), 5)),
        (s('WGT') * s('PTVV'), True),
    )


def test_cleanup_model(pheno):
    model = cleanup_model(pheno)
    assert model.statements.after_odes[1].symbol != s('W')


def test_greekify_model(pheno):
    model = cleanup_model(pheno)
    model = greekify_model(model)
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
        (
            'nonmem/modeling/transit_indirect_reabsorption.mod',
            'absorption',
            ['K12', 'K23', 'KA', 'K45'],
        ),
        (
            'nonmem/modeling/transit_indirect_reabsorption.mod',
            'distribution',
            ['V'],
        ),
        ('nonmem/modeling/transit_indirect_reabsorption.mod', 'elimination', ['CL']),
    ),
    ids=repr,
)
def test_get_pk_parameters(load_model_for_test, testdata, model_path, kind, expected):
    model = load_model_for_test(testdata / model_path)
    assert set(get_pk_parameters(model, kind)) == set(expected)

    try:
        pkpd_model = add_effect_compartment(model, "linear")
    except ValueError:  # Model couldn't be transformed
        pass
    else:
        assert set(get_pk_parameters(pkpd_model, kind)) == set(expected)


def test_get_pk_parameters_metabolite(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno.mod")
    model = add_metabolite(model)

    expected = ['CL', 'CLM', 'V', 'VM']

    assert set(get_pk_parameters(model)) == set(expected)

    model = add_peripheral_compartment(model, "METABOLITE")

    expected.extend(['QP1', 'VP1'])
    assert set(get_pk_parameters(model)) == set(expected)


@pytest.mark.parametrize(
    ('model_path', 'kind', 'expected'),
    (
        # ('nonmem/pheno.mod', 'baseline', ['B']),
        ('nonmem/pheno.mod', 'linear', ['B', 'SLOPE']),
        ('nonmem/pheno.mod', 'emax', ['B', 'E_MAX', 'EC_50']),
        ('nonmem/pheno.mod', 'step', ['B', 'E_MAX']),
        ('nonmem/pheno.mod', 'sigmoid', ['B', 'EC_50', 'E_MAX', 'N']),
    ),
    ids=repr,
)
def test_get_pd_parameters(load_model_for_test, testdata, model_path, kind, expected):
    model = load_model_for_test(testdata / model_path)
    assert set(get_pd_parameters(set_direct_effect(model, kind))) == set(expected)
    assert set(get_pd_parameters(add_effect_compartment(model, kind))) == set(expected + ["MET"])
    assert get_pk_parameters(add_effect_compartment(model, kind)) == ['CL', 'V']
    assert get_pk_parameters(set_direct_effect(model, kind)) == ['CL', 'V']
    assert not set(
        set(get_pd_parameters(set_direct_effect(model, kind))).intersection(
            get_pk_parameters(set_direct_effect(model, kind))
        )
    )
    assert not set(
        set(get_pd_parameters(add_effect_compartment(model, kind))).intersection(
            get_pk_parameters(add_effect_compartment(model, kind))
        )
    )


@pytest.mark.parametrize(
    ('model_path', 'kind', 'prod', 'expected'),
    (
        ('nonmem/pheno.mod', 'linear', True, ['B', 'SLOPE', 'MET']),
        ('nonmem/pheno.mod', 'linear', False, ['B', 'SLOPE', 'MET']),
        ('nonmem/pheno.mod', 'emax', True, ['B', 'E_MAX', 'EC_50', 'MET']),
        ('nonmem/pheno.mod', 'emax', False, ['B', 'E_MAX', 'EC_50', 'MET']),
        ('nonmem/pheno.mod', 'sigmoid', True, ['B', 'EC_50', 'E_MAX', 'N', 'MET']),
        ('nonmem/pheno.mod', 'sigmoid', False, ['B', 'EC_50', 'E_MAX', 'N', 'MET']),
    ),
    ids=repr,
)
def test_get_pd_parameters_indirect(
    load_model_for_test, testdata, model_path, kind, prod, expected
):
    model = load_model_for_test(testdata / model_path)
    assert set(get_pd_parameters(add_indirect_effect(model, kind, prod=prod))) == set(expected)
    assert get_pk_parameters(add_indirect_effect(model, kind, prod=prod)) == ['CL', 'V']
    assert not set(
        set(get_pd_parameters(add_indirect_effect(model, kind, prod=prod))).intersection(
            get_pk_parameters(add_indirect_effect(model, kind, prod=prod))
        )
    )


@pytest.mark.parametrize(
    ('model_path', 'level', 'expected'),
    (
        ('nonmem/pheno.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno.mod', 'random', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno.mod', 'iov', []),
        ('nonmem/pheno_real.mod', 'all', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'random', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'iiv', ['CL', 'V']),
        ('nonmem/pheno_real.mod', 'iov', []),
        ('nonmem/models/mox1.mod', 'all', ['CL', 'KA', 'V']),
        ('nonmem/models/mox1.mod', 'random', ['CL', 'KA', 'V']),
        ('nonmem/models/mox1.mod', 'iiv', ['CL', 'KA', 'V']),
        ('nonmem/models/mox1.mod', 'iov', []),
        ('nonmem/models/mox2.mod', 'all', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox2.mod', 'random', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox2.mod', 'iiv', ['CL', 'MAT', 'VC']),
        ('nonmem/models/mox2.mod', 'iov', []),
        ('nonmem/models/mox_2comp.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/mox_2comp.mod', 'random', ['K', 'K12', 'K21']),
        ('nonmem/models/mox_2comp.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/mox_2comp.mod', 'iov', []),
        ('nonmem/models/pef.mod', 'all', ['K']),
        ('nonmem/models/pef.mod', 'random', ['K']),
        ('nonmem/models/pef.mod', 'iiv', ['K']),
        ('nonmem/models/pef.mod', 'iov', []),
        ('nonmem/models/pheno_advan3_trans1.mod', 'all', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'random', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iiv', ['K', 'K12', 'K21']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'iov', []),
        ('nonmem/models/pheno_noifs.mod', 'random', ['CL', 'V']),
        ('nonmem/models/pheno_noifs.mod', 'all', ['CL', 'V']),
        ('nonmem/models/pheno_noifs.mod', 'iiv', ['CL', 'V']),
        ('nonmem/models/pheno_noifs.mod', 'iov', []),
        ('nonmem/models/pheno_trans1.mod', 'all', ['K']),
        ('nonmem/models/pheno_trans1.mod', 'random', ['K']),
        ('nonmem/models/pheno_trans1.mod', 'iiv', ['K']),
        ('nonmem/models/pheno_trans1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run1.mod', 'all', ['CL', 'V']),
        ('nonmem/secondary_parameters/run1.mod', 'random', ['CL', 'V']),
        ('nonmem/secondary_parameters/run1.mod', 'iiv', ['CL', 'V']),
        ('nonmem/secondary_parameters/run1.mod', 'iov', []),
        ('nonmem/secondary_parameters/run2.mod', 'all', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'random', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'iiv', ['CL', 'V']),
        ('nonmem/secondary_parameters/run2.mod', 'iov', []),
        ('nonmem/qa/iov.mod', 'random', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'random', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'iiv', ['CL', 'V']),
        ('nonmem/qa/iov.mod', 'iov', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'all', ['CL', 'V', 'K23', 'K34', 'K43', 'K12']),
        ('nonmem/modeling/pheno_1transit.mod', 'random', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_1transit.mod', 'iov', []),
        (
            'nonmem/modeling/pheno_2transits.mod',
            'all',
            ['CL', 'V', 'K34', 'K45', 'K54', 'K12', 'K23'],
        ),
        ('nonmem/modeling/pheno_2transits.mod', 'random', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iiv', ['CL', 'V']),
        ('nonmem/modeling/pheno_2transits.mod', 'iov', []),
    ),
    ids=repr,
)
def test_get_individual_parameters(load_model_for_test, testdata, model_path, level, expected):
    model = load_model_for_test(testdata / model_path)
    assert set(get_individual_parameters(model, level)) == set(expected)


def test_get_individual_parameters_redundant_assign(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_first_order_absorption(model)
    model = set_transit_compartments(model, 3)

    assert set(get_individual_parameters(model)) == {'CL', 'MAT', 'MDT', 'V'}


basic_pk_model = create_basic_pk_model()


@pytest.mark.parametrize(
    ('func', 'level', 'dv', 'expected'),
    (
        (partial(set_direct_effect, expr='linear'), 'all', None, ['B', 'CL', 'SLOPE', 'VC']),
        (partial(set_direct_effect, expr='linear'), 'all', 1, ['CL', 'VC']),
        (partial(set_direct_effect, expr='linear'), 'all', 2, ['B', 'CL', 'SLOPE', 'VC']),
        (partial(set_direct_effect, expr='emax'), 'all', None, ['B', 'CL', 'EC_50', 'E_MAX', 'VC']),
        (
            partial(set_direct_effect, expr='sigmoid'),
            'all',
            None,
            ['B', 'CL', 'EC_50', 'E_MAX', 'N', 'VC'],
        ),
        (
            partial(add_effect_compartment, expr='sigmoid'),
            'all',
            None,
            ['B', 'CL', 'EC_50', 'E_MAX', 'MET', 'N', 'VC'],
        ),
        (
            partial(add_indirect_effect, expr='linear'),
            'all',
            None,
            ['B', 'CL', 'MET', 'SLOPE', 'VC'],
        ),
        (
            partial(add_indirect_effect, expr='sigmoid'),
            'all',
            None,
            ['B', 'CL', 'EC_50', 'E_MAX', 'MET', 'N', 'VC'],
        ),
    ),
)
def test_get_individual_parameters_pkpd_models(func, level, dv, expected):
    model = func(basic_pk_model)
    params = get_individual_parameters(model, level=level, dv=dv)
    assert params == expected


@pytest.mark.parametrize(
    ('func', 'level', 'dv', 'expected'),
    (
        (add_metabolite, 'all', None, ['CL', 'CLM', 'VC', 'VM']),
        (
            partial(add_metabolite, presystemic=True),
            'all',
            None,
            ['CL', 'CLM', 'FPRE', 'MAT', 'VC', 'VM'],
        ),
    ),
)
def test_get_individual_parameters_drug_metabolite_models(func, level, dv, expected):
    model = func(basic_pk_model)
    params = get_individual_parameters(model, level=level, dv=dv)
    assert params == expected


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
def test_has_random_effect(load_model_for_test, testdata, model_path, level, expected):
    model = load_model_for_test(testdata / model_path)
    params_with_random_effect = set(expected)
    for param in get_pk_parameters(model):
        if param in params_with_random_effect:
            assert has_random_effect(model, param, level)
        else:
            assert not has_random_effect(model, param, level)


@pytest.mark.parametrize(
    ('model_path', 'rv', 'expected'),
    (
        ('nonmem/pheno.mod', 'ETA_1', ['CL']),
        ('nonmem/pheno.mod', 'ETA_2', ['V']),
        ('nonmem/pheno_real.mod', 'ETA_1', ['CL']),
        ('nonmem/pheno_real.mod', 'ETA_2', ['V']),
        ('nonmem/models/mox1.mod', 'ETA_1', ['CL']),
        ('nonmem/models/mox1.mod', 'ETA_2', ['V']),
        ('nonmem/models/mox1.mod', 'ETA_3', ['KA']),
        ('nonmem/models/mox2.mod', 'ETA_1', ['CL']),
        ('nonmem/models/mox2.mod', 'ETA_2', ['VC']),
        ('nonmem/models/mox2.mod', 'ETA_3', ['MAT']),
        ('nonmem/models/mox_2comp.mod', 'ETA_1', ['K12']),
        ('nonmem/models/mox_2comp.mod', 'ETA_2', ['K21']),
        ('nonmem/models/mox_2comp.mod', 'ETA_3', ['K']),
        ('nonmem/models/pef.mod', 'ETA_1', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'ETA_1', ['K']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'ETA_2', ['K12']),
        ('nonmem/models/pheno_advan3_trans1.mod', 'ETA_3', ['K21']),
        ('nonmem/models/pheno_noifs.mod', 'ETA_1', ['CL']),
        ('nonmem/models/pheno_noifs.mod', 'ETA_2', ['V']),
        ('nonmem/models/pheno_trans1.mod', 'ETA_1', ['K']),
        ('nonmem/secondary_parameters/run1.mod', 'ETA_1', ['CL']),
        ('nonmem/secondary_parameters/run1.mod', 'ETA_2', ['V']),
        ('nonmem/secondary_parameters/run2.mod', 'ETA_1', ['CL']),
        ('nonmem/secondary_parameters/run2.mod', 'ETA_2', ['V']),
        ('nonmem/qa/iov.mod', 'ETA_1', ['CL']),
        ('nonmem/qa/iov.mod', 'ETA_2', ['V']),
        ('nonmem/qa/iov.mod', 'ETA_3', ['CL']),
        ('nonmem/qa/iov.mod', 'ETA_4', ['V']),
        ('nonmem/qa/iov.mod', 'ETA_5', ['CL']),
        ('nonmem/qa/iov.mod', 'ETA_6', ['V']),
        ('nonmem/modeling/pheno_1transit.mod', 'ETA_1', ['CL']),
        ('nonmem/modeling/pheno_1transit.mod', 'ETA_2', ['V']),
        (
            'nonmem/modeling/pheno_2transits.mod',
            'ETA_1',
            ['CL'],
        ),
        ('nonmem/modeling/pheno_2transits.mod', 'ETA_2', ['V']),
    ),
    ids=repr,
)
def test_get_rv_parameter(load_model_for_test, testdata, model_path, rv, expected):
    model = load_model_for_test(testdata / model_path)
    rv_params = get_rv_parameters(model, rv)

    assert rv_params == expected


@pytest.mark.parametrize(
    ('model_path', 'param', 'var_type', 'expected'),
    (
        ('nonmem/pheno.mod', 'CL', 'iiv', ['ETA_1']),
        ('nonmem/pheno_real.mod', 'CL', 'iiv', ['ETA_1']),
        ('nonmem/pheno_real.mod', 'TAD', 'iiv', []),
        ('nonmem/qa/iov.mod', 'CL', 'iiv', ['ETA_1']),
        ('nonmem/qa/iov.mod', 'V', 'iiv', ['ETA_2']),
        ('nonmem/qa/iov.mod', 'CL', 'iiv', ['ETA_1']),
        ('nonmem/qa/iov.mod', 'V', 'iiv', ['ETA_2']),
        ('nonmem/qa/iov.mod', 'CL', 'iov', ['ETA_3', 'ETA_5']),
        ('nonmem/qa/iov.mod', 'V', 'iov', ['ETA_4', 'ETA_6']),
    ),
    ids=repr,
)
def test_get_parameter_rv(load_model_for_test, testdata, model_path, param, var_type, expected):
    model = load_model_for_test(testdata / model_path)
    iiv_names = get_parameter_rv(model, param, var_type)
    assert iiv_names == expected


def test_get_rv_parameter_verify_input(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    with pytest.raises(ValueError, match='Could not find random variable: x'):
        get_rv_parameters(model, 'x')


def test_get_parameter_rv_verify_input(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    with pytest.raises(ValueError, match='Could not find parameter x'):
        get_parameter_rv(model, 'x')
    with pytest.raises(
        ValueError,
        match='ETA_1 is a random variable. Only parameters are accepted as input',
    ):
        get_parameter_rv(model, 'ETA_1')


def test_display_odes(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    display_odes(model)


def test_is_real(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert is_real(model, "CL")
    assert not is_real(model, "I*CL")
    assert is_real(model, "sqrt(CL)") is None


def test_is_linearized(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    assert not is_linearized(model)
    model = load_example_model_for_test("pheno_linear")
    assert is_linearized(model)


def test_get_dv_symbol(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    assert get_dv_symbol(model, 2) == Expr.symbol("Y_2")
    assert get_dv_symbol(model, "Y_1") == Expr.symbol("Y_1")
    assert get_dv_symbol(model, Expr.symbol("Y_2")) == Expr.symbol("Y_2")
    assert get_dv_symbol(model) == Expr.symbol("Y_1")
    assert get_dv_symbol(model, None) == Expr.symbol("Y_1")

    with pytest.raises(ValueError):
        get_dv_symbol(model, 3)
    with pytest.raises(ValueError):
        get_dv_symbol(model, "FLUMOX")
    with pytest.raises(ValueError):
        get_dv_symbol(model, Expr.symbol("SPANNER"))
    with pytest.raises(ValueError):
        get_dv_symbol(model, 3.4)
