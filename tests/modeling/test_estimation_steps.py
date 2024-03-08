import pytest

from pharmpy.model import SimulationStep
from pharmpy.modeling import (
    add_estimation_step,
    add_parameter_uncertainty_step,
    add_predictions,
    add_residuals,
    append_estimation_step_options,
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
    assert model.model_code.split('\n')[-2] == code_ref


def test_set_estimation_step_est_middle(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_estimation_step(
        model, 'FOCE', interaction=True, parameter_uncertainty_method='SANDWICH', idx=0
    )
    assert (
        '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC\n$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
        in model.model_code
    )


def test_add_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_estimation_step(model, 'fo')
    assert len(model.estimation_steps) == 2
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO'

    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_estimation_step(model, 'fo', evaluation=True)
    assert len(model.estimation_steps) == 2
    assert model.model_code.split('\n')[-2] == '$ESTIMATION METHOD=ZERO MAXEVAL=0'


def test_add_estimation_step_non_int(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_estimation_step(model, 'fo', idx=1.0)
    with pytest.raises(TypeError, match='Index must be integer'):
        add_estimation_step(model, 'fo', idx=1.5)


def test_remove_estimation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = remove_estimation_step(model, 0)
    assert not model.estimation_steps
    assert model.model_code.split('\n')[-2] == '$SIGMA 1'


def test_add_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert len(model.estimation_steps) == 1
    assert model.model_code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert len(model.estimation_steps) == 1
    assert (
        model.model_code.split('\n')[-2] == '$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1'
    )

    model = remove_parameter_uncertainty_step(model)

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.estimation_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.model_code
    )


def test_remove_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = add_parameter_uncertainty_step(model, 'SANDWICH')
    assert model.model_code.split('\n')[-2] == '$COVARIANCE UNCONDITIONAL PRINT=E PRECOND=1'
    model = remove_parameter_uncertainty_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )

    model = add_parameter_uncertainty_step(model, "EFIM")
    assert len(model.estimation_steps) == 1
    assert (
        "$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA file.csv IGNORE=@ REWIND\n"
        "$INPUT ID DV TIME\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.model_code
    )
    model = remove_parameter_uncertainty_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC'
    )


def test_parse_parameter_uncertainty_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_design.mod')
    assert (
        "$ESTIMATION METHOD=1 INTERACTION MSFO=pheno_design.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA pheno.dta IGNORE=@ REWIND\n"
        "$INPUT ID TIME AMT WGT APGR DV\n"
        "$MSFI pheno_design.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n" in model.model_code
    )
    assert model.estimation_steps[-1].tool_options == {}


def test_update_parameter_uncertainty_method(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert (
        "$ESTIMATION METHOD=COND INTER\n" "$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1\n"
    ) in model.model_code
    assert model.estimation_steps[-1].parameter_uncertainty_method == 'SMAT'
    model = add_parameter_uncertainty_step(model, 'EFIM')
    assert (
        "$ESTIMATION METHOD=COND INTER MSFO=efim.msf\n"
        "$PROBLEM DESIGN\n"
        "$DATA 'pheno.dta' IGNORE=@ REWIND\n"
        "$INPUT ID TIME AMT WGT APGR DV FA1 FA2\n"
        "$MSFI efim.msf\n"
        "$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n"
    ) in model.model_code
    assert model.estimation_steps[-1].parameter_uncertainty_method == 'EFIM'
    model = remove_parameter_uncertainty_step(model)
    assert ("$ESTIMATION METHOD=COND INTER\n") in model.model_code
    assert model.estimation_steps[-1].parameter_uncertainty_method is None
    model = add_parameter_uncertainty_step(model, 'SMAT')
    assert (
        "$ESTIMATION METHOD=COND INTER\n" "$COVARIANCE MATRIX=S UNCONDITIONAL PRINT=E PRECOND=1\n"
    ) in model.model_code
    assert model.estimation_steps[-1].parameter_uncertainty_method == 'SMAT'


def test_append_estimation_step_options(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    assert len(model.estimation_steps) == 1
    model = append_estimation_step_options(model, {'SADDLE_RESET': 1}, 0)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=9990 PRINT=2 POSTHOC SADDLE_RESET=1'
    )


def test_set_evaluation_step(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_evaluation_step(model)
    assert (
        model.model_code.split('\n')[-2]
        == '$ESTIMATION METHOD=COND INTER MAXEVAL=0 PRINT=2 POSTHOC'
    )


def test_set_simulation(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'minimal.mod')
    model = set_simulation(model, n=2, seed=1234)
    assert len(model.estimation_steps) == 1
    assert model.estimation_steps[0] == SimulationStep(n=2, seed=1234)
    assert model.model_code.split('\n')[-2] == "$SIMULATION (1234) SUBPROBLEMS=2"


def test_add_predictions(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    assert model.estimation_steps[-1].predictions == ('IPRED', 'PRED')

    model = add_predictions(model, pred=['PRED1'])
    assert model.estimation_steps[-1].predictions == ('IPRED', 'PRED', 'PRED1')
    assert tuple(sorted(model.estimation_steps[-1].residuals)) == ('CWRES', 'RES')

    model = add_residuals(model, res=['RES', 'RES2'])
    assert model.estimation_steps[-1].residuals == ('CWRES', 'RES', 'RES2')

    model = add_predictions(model, pred=['PRED1', 'PRED2'])
    assert model.estimation_steps[-1].predictions == ('IPRED', 'PRED', 'PRED1', 'PRED2')
    assert model.estimation_steps[-1].residuals == ('CWRES', 'RES', 'RES2')
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
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE PRED1 RES2 PRED2 NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.model_code
    model = remove_predictions(model, 'all')
    model = remove_residuals(model, 'all')
    assert model.estimation_steps[-1].predictions == ()
    assert model.estimation_steps[-1].residuals == ()
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
$TABLE ID TIME DV AMT WGT APGR TAD NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.model_code
    model = add_residuals(model, res=['NEWRES'])
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
$TABLE ID TIME DV AMT WGT APGR TAD NPDE NEWRES NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1\n"""
    assert model_code == model.model_code
    model = remove_residuals(model, ['NEWRES'])
    assert model.estimation_steps[-1].residuals == ()

    # Test $DESIGN
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_design.mod')
    assert model.estimation_steps[-1].predictions == ()
    assert model.estimation_steps[-1].residuals == ()
    model = add_predictions(model, pred=['PRED1'])
    assert model.estimation_steps[-1].predictions == ('PRED1',)
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
$TABLE ID TIME DV PRED1 FILE=mytab NOAPPEND NOPRINT
$DESIGN APPROX=FO FIMDIAG=1 GROUPSIZE=1 OFVTYPE=1\n"""
    assert model_code == model.model_code
    model = add_residuals(model, res=['RES'])
    model = remove_predictions(model, 'all')
    assert model.estimation_steps[-1].predictions == ()
    assert model.estimation_steps[-1].residuals == ('RES',)
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
$TABLE ID TIME DV RES FILE=mytab NOAPPEND NOPRINT\n"""
    assert model_code == model.model_code
