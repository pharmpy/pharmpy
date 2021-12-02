from io import StringIO

from pyfakefs.fake_filesystem_unittest import Patcher

from pharmpy import Model
from pharmpy.modeling import (
    has_additive_error_model,
    has_combined_error_model,
    has_proportional_error_model,
    load_example_model,
    read_model_from_string,
    remove_error_model,
    set_additive_error_model,
    set_combined_error_model,
    set_dtbs_error_model,
    set_proportional_error_model,
    set_weighted_error_model,
    use_thetas_for_error_stdev,
)
from pharmpy.modeling.error import _get_prop_init
from pharmpy.statements import Assignment


def test_remove_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    remove_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F'


def test_set_additive_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    set_additive_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F + EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA  11.2225 ; sigma'
    before = model.model_code
    set_additive_error_model(model)  # One more time and nothing should change
    assert before == model.model_code


def test_set_additive_error_model_logdv(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    set_additive_error_model(model, data_trans="log(Y)")
    assert model.model_code.split('\n')[11] == 'Y = LOG(F) + EPS(1)/F'
    assert model.model_code.split('\n')[17] == '$SIGMA  11.2225 ; sigma'


def test_set_proportional_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    model.statements[5] = Assignment('Y', 'F')
    set_proportional_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y=F+F*EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA  0.09 ; sigma'

    model = Model(testdata / 'nonmem' / 'pheno.mod')
    set_proportional_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y=F+F*EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA 0.013241'


def test_set_proportional_error_model_log(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    model.statements[5] = Assignment('Y', 'F')
    set_proportional_error_model(model, data_trans='log(Y)')
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
IF (F.EQ.0) THEN
    IPREDADJ = 2.22500000000000E-307
ELSE
    IPREDADJ = F
END IF
Y = LOG(IPREDADJ) + EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA  0.09 ; sigma
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_combined_error_model(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    set_combined_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F + EPS(1)*F + EPS(2)'
    assert model.model_code.split('\n')[17] == '$SIGMA  0.09 ; sigma_prop'
    assert model.model_code.split('\n')[18] == '$SIGMA  11.2225 ; sigma_add'
    before = model.model_code
    set_combined_error_model(model)  # One more time and nothing should change
    assert before == model.model_code


def test_set_combined_error_model_log(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    set_combined_error_model(model, data_trans='log(Y)')
    assert model.model_code.split('\n')[11] == 'Y = LOG(F) + EPS(2)/F + EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA  0.09 ; sigma_prop'
    assert model.model_code.split('\n')[18] == '$SIGMA  11.2225 ; sigma_add'


def test_remove_error_without_f(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
Y=CONC+CONC*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model(StringIO(code))
    remove_error_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
Y = CONC
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_additive_error_without_f(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y=CONC+CONC*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_additive_error_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y = CONC + EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA  11.2225 ; sigma
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_get_prop_init(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')

    init = _get_prop_init(model.dataset)
    assert init == 11.2225

    model.dataset['DV'].values[:] = 0.0
    init = _get_prop_init(model.dataset)
    assert init == 0.01


def test_has_additive_error_model(testdata):
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert has_additive_error_model(model)
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
Y = (CONC + ETA(1)) + (CONC + ETA(1)) * ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_additive_error_model(model)

    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
Y = CONC + EPS(1) + CONC*EPS(2)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_additive_error_model(model)

    model = Model(StringIO(code))
    assert not has_additive_error_model(model)


def test_has_proportional_error_model(testdata):
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_proportional_error_model(model)
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
Y = (CONC + ETA(1)) + (CONC + ETA(1)) * ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert has_proportional_error_model(model)

    model = Model(StringIO(code))
    assert has_proportional_error_model(model)


def test_has_combined_error_model(testdata):
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_combined_error_model(model)
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
Y = (CONC + ETA(1)) + (CONC + ETA(1)) * ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_combined_error_model(model)

    model = Model(StringIO(code))
    assert not has_combined_error_model(model)

    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
Y = (CONC + ETA(1)) + (CONC + ETA(1)) * ERR(1) + ERR(2)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$SIGMA .02
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert has_combined_error_model(model)


def test_use_theta_for_error_stdev():
    code = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + ERR(1)

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    use_thetas_for_error_stdev(model)
    correct = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)*THETA(2)

$THETA 0.1
$THETA  (0,1.0) ; SD_EPS(1)
$OMEGA 0.01
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""

    assert model.model_code == correct


def test_set_weighted_error_model():
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y=CONC+CONC*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    set_weighted_error_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
W = CONC
Y = CONC + EPS(1)*W
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""

    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
Y=CONC+CONC*THETA(3)*EPS(1)+THETA(4)*EPS(2)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA (0, 0.1) ; SD1
$THETA (0, 0.2) ; SD2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    set_weighted_error_model(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
PRED=A(1)/V
CONC=PRED
W = SQRT(CONC**2*THETA(3)**2 + THETA(4)**2)
Y = CONC + EPS(1)*W
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA (0, 0.1) ; SD1
$THETA (0, 0.2) ; SD2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_dtbs_error_model():
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
Y=CONC+CONC*EPS(1)+EPS(2)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    model.name = 'run1'
    set_dtbs_error_model(model)

    with Patcher(additional_skip_names=['pkgutil']):
        model.update_source()
        with open('run1_contr.f90') as fh:
            assert fh.readline().startswith('      subroutine contr')
        with open('run1_ccontra.f90') as fh:
            assert fh.readline().startswith('      subroutine ccontr')

    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
$ERROR
CONC=A(1)/V
W = SQRT(CONC**2*THETA(3)**2 + THETA(4)**2)
W = CONC**THETA(6)*W
IF (CONC.NE.0.AND.THETA(5).NE.0) THEN
    IPRED = (CONC**THETA(5) - 1)/THETA(5)
ELSE IF (THETA(5).EQ.0.AND.CONC.NE.0) THEN
    IPRED = LOG(CONC)
ELSE IF (CONC.EQ.0.AND.THETA(5).EQ.0) THEN
    IPRED = -1/THETA(5)
ELSE
    IPRED = -1000000000
END IF
Y = IPRED + EPS(1)*W
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,0.11506954418958998) ; SD_EPS(1)
$THETA  (0,0.11506954418958998) ; SD_EPS(2)
$THETA  1 ; tbs_lambda
$THETA  0.001 ; tbs_zeta
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
