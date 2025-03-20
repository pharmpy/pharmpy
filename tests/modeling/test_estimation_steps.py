import pytest

from pharmpy.model import SimulationStep
from pharmpy.modeling import (
    add_derivative,
    add_estimation_step,
    add_parameter_uncertainty_step,
    add_predictions,
    add_residuals,
    append_estimation_step_options,
    is_simulation_model,
    remove_derivative,
    remove_estimation_step,
    remove_parameter_uncertainty_step,
    remove_predictions,
    remove_residuals,
    set_estimation_step,
    set_evaluation_step,
    set_simulation,
)


@pytest.mark.parametrize(
    'method,kwargs,code_ref',
    [
        (
            'fo',
            {'interaction': False},
            '$ESTIMATION METHOD=ZERO MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'interaction': True},
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'tool_options': {'saddle_reset': 1}},
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9990 PRINT=2 SADDLE_RESET=1',
        ),
        (
            'bayes',
            {'interaction': True},
            '$ESTIMATION METHOD=BAYES INTER MAXEVAL=9990 PRINT=2 POSTHOC',
        ),
        (
            'fo',
            {'interaction': False, 'evaluation': True, 'maximum_evaluations': None},
            '$ESTIMATION METHOD=ZERO MAXEVAL=0 PRINT=2 POSTHOC',
        ),
    ],
)
def test_set_estimation_step(testdata, load_model_for_test, method, kwargs, code_ref):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_estimation_step(model, method, **kwargs)
    assert model.code.split('\n')[-2] == code_ref


def test_set_estimation_step_est_middle(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_estimation_step(
        model, 'FOCE', interaction=True, parameter_uncertainty_method='SANDWICH', idx=0
    )
    assert (
        '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC\n$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
        in model.code
    )


def test_add_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.execution_steps) == 1
    model = add_estimation_step(model, 'fo')
    assert len(model.execution_steps) == 2
    assert model.code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO'

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.execution_steps) == 1
    model = add_estimation_step(model, 'fo', evaluation=True)
    assert len(model.execution_steps) == 2
    assert model.code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'


def test_add_estimation_step_non_int(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_estimation_step(model, 'fo', idx=1.0)
    with pytest.raises(TypeError, match='Index must be integer'):
        add_estimation_step(model, 'fo', idx=1.5)


def test_remove_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.execution_steps) == 1
    model = remove_estimation_step(model, 0)
    assert not model.execution_steps
    assert model.code.split('\n')[-2] == '$SIGMA 1'


def test_add_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.execution_steps) == 1
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert len(model.execution_steps) == 1
    assert model.code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert len(model.execution_steps) == 1
    assert model.code.split('\n')[-2] == '$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1'

    model = remove_parameter_uncertainty_step(model)

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.execution_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.code
    )


def test_remove_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert model.code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    assert (
        model.code.split('\n')[-2] == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.execution_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.code
    )
    model = remove_parameter_uncertainty_step(model)
    assert (
        model.code.split('\n')[-2] == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )


def test_parse_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_design.mod')
    assert (
        "$ESTIMATION METHOD=1 INTERACTION MSFO=pheno_design.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA pheno.dta IGNORE=@ REWIND\n"
        "$INPUT ID TIME AMT WGT APGR DV\n"
        "$MSFI pheno_design.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.code
    )
    assert model.execution_steps[-1].tool_options == {}


def test_update_parameter_uncertainty_method(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert (
        "$ESTIMATION METHOD=COND INTER\n" "$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1\n"
    ) in model.code
    assert model.execution_steps[-1].parameter_uncertainty_method == 'SMAT'
    model = add_parameter_uncertainty_step(model, 'EFIM')
    assert (
        "$ESTIMATION METHOD=COND INTER MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA 'pheno.dta' IGNORE=@ REWIND\n"
        "$INPUT ID TIME AMT WGT APGR DV FA1 FA2\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n"
    ) in model.code
    assert model.execution_steps[-1].parameter_uncertainty_method == 'EFIM'
    model = remove_parameter_uncertainty_step(model)
    assert ("$ESTIMATION METHOD=COND INTER\n") in model.code
    assert model.execution_steps[-1].parameter_uncertainty_method is None
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert (
        "$ESTIMATION METHOD=COND INTER\n" "$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1\n"
    ) in model.code
    assert model.execution_steps[-1].parameter_uncertainty_method == 'SMAT'


def test_append_estimation_step_options(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.execution_steps) == 1
    model = append_estimation_step_options(model, {'SADDLE_RESET': 1}, 0)
    assert (
        model.code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC SADDLE_RESET=1'
    )


def test_set_evaluation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_evaluation_step(model)
    assert model.code.split('\n')[-2] == '$ESTIMATION METHOD=COND INTER MAXEVAL=0 PRINT=2 POSTHOC'


def test_set_simulation(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_simulation(model, n=2, seed=1234)
    assert len(model.execution_steps) == 1
    assert model.execution_steps[0] == SimulationStep(n=2, seed=1234)
    assert model.code.split('\n')[-2] == "$SIMULATION (1234) SUBPROBLEMS=2 ONLYSIMULATION"


def test_add_predictions_raise(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    with pytest.raises(ValueError, match="Prediction variables need to be one of the following:"):
        model = add_predictions(model, pred=["NOT_A_PREDICTION"])


def test_add_residuals_raise(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    with pytest.raises(ValueError, match="Residual variables need to be one of the following"):
        model = add_residuals(model, res=["NOT_A_RESIDUAL"])


def test_add_predictions(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    assert model.execution_steps[-1].predictions == ('IPRED', 'PRED')

    model = add_predictions(model, pred=['CIPREDI'])
    assert model.execution_steps[-1].predictions == ('CIPREDI', 'IPRED', 'PRED')
    assert tuple(sorted(model.execution_steps[-1].residuals)) == ('CWRES', 'RES')

    model = add_residuals(model, res=['RES', 'IRES'])
    assert model.execution_steps[-1].residuals == ('CWRES', 'IRES', 'RES')

    model = add_predictions(model, pred=['CIPREDI'])
    assert model.execution_steps[-1].predictions == ('CIPREDI', 'IPRED', 'PRED')
    assert model.execution_steps[-1].residuals == ('CWRES', 'IRES', 'RES')
    model_code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
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
$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE
 CIPREDI IRES NOAPPEND NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.code
    model = remove_predictions(model)
    model = remove_residuals(model)
    assert model.execution_steps[-1].predictions == ()
    assert model.execution_steps[-1].residuals == ()
    model_code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
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
$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1
$TABLE ID TIME DV AMT WGT APGR TAD NPDE
 NOAPPEND NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.code
    model = add_residuals(model, res=['CWRES'])
    model_code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
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
$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1
$TABLE ID TIME DV AMT WGT APGR TAD NPDE
 CWRES NOAPPEND NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.code
    model = remove_residuals(model, ['CWRES'])
    assert model.execution_steps[-1].residuals == ()

    # Test $DESIGN
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_design.mod')
    assert model.execution_steps[-1].predictions == ()
    assert model.execution_steps[-1].residuals == ()
    model = add_predictions(model, pred=['IPRED'])
    assert model.execution_steps[-1].predictions == ('IPRED',)
    model_code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION MSFO=pheno_design.msf
$PROBLEM DESIGN
$DATA pheno.dta IGNORE=@ REWIND
$INPUT ID TIME AMT WGT APGR DV
$MSFI pheno_design.msf
$TABLE ID TIME DV IPRED FILE=mytabb ONEHEADER NOAPPEND NOPRINT
$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n"""
    assert model_code == model.code
    model = add_residuals(model, res=['RES'])
    model = remove_predictions(model)
    assert model.execution_steps[-1].predictions == ()
    assert model.execution_steps[-1].residuals == ('RES',)
    model_code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION MSFO=pheno_design.msf
$PROBLEM DESIGN
$DATA pheno.dta IGNORE=@ REWIND
$INPUT ID TIME AMT WGT APGR DV
$MSFI pheno_design.msf
$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1
$TABLE ID TIME DV RES FILE=mytabb ONEHEADER NOAPPEND NOPRINT\n"""
    assert model_code == model.code


def test_add_remove_derivative(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')

    model = add_derivative(model)
    assert len(model.execution_steps[0].derivatives) == 5
    model = add_derivative(model)
    assert len(model.execution_steps[0].derivatives) == 5

    assert (
        model.code
        == """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W
D_EPSETA_1_1 = 0
D_EPSETA_1_2 = 0

"LAST
"  D_EPSETA_1_1=HH(1, 2)
"  D_EPSETA_1_2=HH(1, 3)
$THETA (0,0.00469307) ; PTVCL
$THETA (0,1.00916) ; PTVV
$THETA (-.99,.1)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 0.013241
$ESTIMATION METHOD=COND INTER
$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE
 G011 G021 H011 D_EPSETA_1_1 D_EPSETA_1_2 NOAPPEND NOPRINT ONEHEADER FILE=sdtab1 RFORMAT="(1PE16.9,300(1PE24.16))"\n"""
    )

    model = remove_derivative(model)
    assert len(model.execution_steps[0].derivatives) == 0
    model = remove_derivative(model)
    assert len(model.execution_steps[0].derivatives) == 0

    model = add_derivative(model, "ETA_1")
    assert len(model.execution_steps[0].derivatives) == 1
    model = add_derivative(model, (("ETA_1", "ETA_2"),))
    assert len(model.execution_steps[0].derivatives) == 2
    model = add_derivative(model, (("ETA_1", "EPS_1"), "EPS_1"))
    assert len(model.execution_steps[0].derivatives) == 4

    model = remove_derivative(model, "ETA_1")
    assert len(model.execution_steps[0].derivatives) == 3
    model = remove_derivative(model, (("ETA_1", "ETA_2"),))
    assert len(model.execution_steps[0].derivatives) == 2
    model = remove_derivative(model, (("ETA_1", "EPS_1"), "EPS_1"))
    assert len(model.execution_steps[0].derivatives) == 0

    assert (
        model.code
        == """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA 'pheno.dta' IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
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
$ESTIMATION METHOD=COND INTER
$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE
 NOAPPEND NOPRINT ONEHEADER FILE=sdtab1 RFORMAT="(1PE16.9,300(1PE24.16))"\n"""
    )


def test_is_simulation_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    assert not is_simulation_model(model)
    m2 = set_simulation(model)
    assert is_simulation_model(m2)
