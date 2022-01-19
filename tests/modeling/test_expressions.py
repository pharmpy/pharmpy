import sympy

from pharmpy import Model
from pharmpy.modeling import (
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    get_individual_prediction_expression,
    get_observation_expression,
    get_population_prediction_expression,
    mu_reference_model,
    read_model,
    read_model_from_string,
)
from pharmpy import RandomVariable, RandomVariables, ModelStatements, Assignment, Parameters, Parameter
import pytest


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
"""
    assert model.model_code == correct


@pytest.mark.usefixtures('testdata')
@pytest.mark.parametrize(
    'statements,correct',
    [
        ([Assignment('CL', s('THETA(1)') + s('ETA(1)'))], [Assignment('mu_1', s('THETA(1)')), Assignment('CL', s('mu_1') + s('ETA(1)'))]),
        ([Assignment('CL', s('THETA(1)')*s('AGE')**s('THETA(2)')*sympy.exp(s('ETA(1)'))), Assignment('V', s('THETA(3)')*sympy.exp(s('ETA(2)')))], [Assignment('mu_1', sympy.log(s('THETA(1)') * s('AGE')**s('THETA(2)'))), Assignment('CL', sympy.exp(s('mu_1') + s('ETA(1)'))), Assignment('mu_2', sympy.log(s('THETA(3)'))), Assignment('V', sympy.exp(s('mu_2') + s('ETA(2)')))]),
        ([Assignment('CL', s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))], [Assignment('CL', s('THETA(1)') + s('ETA(1)') + s('ETA(2)'))]),
    ],
)
def test_mu_reference_model_generic(testdata, statements, correct):
    model = Model()
    model.statements = ModelStatements(statements)
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
    assert model.statements == ModelStatements(correct)
