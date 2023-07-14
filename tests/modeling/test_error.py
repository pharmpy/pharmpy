import re
import shutil

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Assignment
from pharmpy.modeling import (
    has_additive_error_model,
    has_combined_error_model,
    has_proportional_error_model,
    has_weighted_error_model,
    read_model_from_string,
    remove_error_model,
    set_additive_error_model,
    set_combined_error_model,
    set_dtbs_error_model,
    set_iiv_on_ruv,
    set_power_on_ruv,
    set_proportional_error_model,
    set_weighted_error_model,
    use_thetas_for_error_stdev,
)
from pharmpy.modeling.error import _get_prop_init, set_time_varying_error_model


def test_remove_error_model(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = remove_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F'


def test_set_additive_error_model(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_additive_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F + EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA  11.2225 ; sigma'
    before = model.model_code
    model = set_additive_error_model(model)  # One more time and nothing should change
    assert before == model.model_code

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    model = set_additive_error_model(model)
    rec = model.internals.control_stream.get_records('ERROR')[0]
    correct = """$ERROR
Y_1 = F + EPS(3)
Y_2 = F + EPS(1)*F + EPS(2)

IF (DVID.EQ.1) THEN
    Y = Y_1
ELSE
    Y = Y_2
END IF
"""
    assert str(rec) == correct


def test_set_additive_error_model_logdv(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_additive_error_model(model, data_trans="log(Y)")
    assert model.model_code.split('\n')[11] == 'Y = LOG(F) + EPS(1)/F'
    assert model.model_code.split('\n')[17] == '$SIGMA  11.2225 ; sigma'


def test_set_proportional_error_model_nolog(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = model.replace(
        statements=model.statements[0:5] + Assignment.create('Y', 'F') + model.statements[6:]
    )
    model = set_proportional_error_model(model)
    assert model.model_code.split('\n')[16] == 'Y = F + EPS(1)*IPREDADJ'
    assert model.model_code.split('\n')[22] == '$SIGMA  0.09 ; sigma'

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_proportional_error_model(model, zero_protection=False)
    assert model.model_code.split('\n')[11] == 'Y=F+F*EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA 0.013241'

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    model = set_proportional_error_model(model, dv=2)
    rec = model.internals.control_stream.get_records('ERROR')[0]
    correct = """$ERROR
Y_1 = F + EPS(1)*F
IF (F.EQ.0) THEN
    IPREDADJ = 2.22500000000000E-16
ELSE
    IPREDADJ = F
END IF
Y_2 = F + EPS(2)*IPREDADJ

IF (DVID.EQ.1) THEN
    Y = Y_1
ELSE
    Y = Y_2
END IF
"""
    assert str(rec) == correct


def test_set_proportional_error_model_log(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = model.replace(
        statements=model.statements[0:5] + Assignment.create('Y', 'F') + model.statements[6:]
    )
    model = set_proportional_error_model(model, data_trans='log(Y)')
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
    IPREDADJ = 2.22500000000000E-16
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


def test_set_combined_error_model(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_combined_error_model(model)
    assert model.model_code.split('\n')[11] == 'Y = F + EPS(1)*F + EPS(2)'
    assert model.model_code.split('\n')[17] == '$SIGMA  0.09 ; sigma_prop'
    assert model.model_code.split('\n')[18] == '$SIGMA  11.2225 ; sigma_add'
    before = model.model_code
    model = set_combined_error_model(model)  # One more time and nothing should change
    assert before == model.model_code


def test_set_combined_error_model_log(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_combined_error_model(model, data_trans='log(Y)')
    assert model.model_code.split('\n')[11] == 'Y = LOG(F) + EPS(2)/F + EPS(1)'
    assert model.model_code.split('\n')[17] == '$SIGMA  0.09 ; sigma_prop'
    assert model.model_code.split('\n')[18] == '$SIGMA  11.2225 ; sigma_add'


def test_set_combined_error_model_with_eta_on_ruv(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_iiv_on_ruv(model)
    model = set_combined_error_model(model)
    assert model.model_code.split('\n')[12] == 'Y = F + EPS(1)*F*EXP(ETA_RV1) + EPS(2)*EXP(ETA_RV1)'


def test_set_combined_error_model_with_time_varying(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_time_varying_error_model(model, cutoff=1.0)
    model = set_combined_error_model(model)
    assert model.model_code.split('\n')[11] == 'IF (TIME.LT.1.0) THEN'
    assert model.model_code.split('\n')[12] == '    Y = F + EPS(1)*F*THETA(3) + EPS(2)*THETA(3)'
    assert model.model_code.split('\n')[14] == '    Y = F + EPS(1)*F + EPS(2)'


def test_set_combined_error_model_with_time_varying_and_eta_on_ruv(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_time_varying_error_model(model, cutoff=1.0)
    model = set_iiv_on_ruv(model)
    model = set_combined_error_model(model)
    assert (
        model.model_code.split('\n')[13]
        == '    Y = F + EPS(1)*F*THETA(3)*EXP(ETA_RV1) + EPS(2)*THETA(3)*EXP(ETA_RV1)'
    )
    assert (
        model.model_code.split('\n')[15]
        == '    Y = F + EPS(1)*F*EXP(ETA_RV1) + EPS(2)*EXP(ETA_RV1)'
    )


def test_set_combined_error_model_multiple_dvs(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    model = set_combined_error_model(model, dv=1)
    rec = model.internals.control_stream.get_records('ERROR')[0]
    correct = """$ERROR
Y_1 = F + EPS(3)*F + EPS(4)
Y_2 = F + EPS(1)*F + EPS(2)

IF (DVID.EQ.1) THEN
    Y = Y_1
ELSE
    Y = Y_2
END IF
"""
    assert str(rec) == correct


def test_remove_error_without_f(create_model_for_test):
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
    model = create_model_for_test(code)
    model = remove_error_model(model)
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


def test_additive_error_without_f(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    model = set_additive_error_model(model)
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


def test_get_prop_init(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')

    init = _get_prop_init(model)
    assert init == 11.2225

    df = model.dataset.copy()
    df['DV'].values[:] = 0.0
    model = model.replace(dataset=df)
    init = _get_prop_init(model)
    assert init == 0.01


def test_has_additive_error_model(create_model_for_test):
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

    model = create_model_for_test(code)
    assert not has_additive_error_model(model)

    code = """$PROBLEM base model
$INPUT ID DV TIME DVID
$DATA file.csv IGNORE=@

$PRED
IF (DVID.EQ.1) THEN
    Y = THETA(1) + ETA(1) + ERR(1)
ELSE
    Y = THETA(1) + THETA(1) * EPS(2)
END IF

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert has_additive_error_model(model)
    assert has_additive_error_model(model, 1)
    assert not has_additive_error_model(model, 2)


def test_has_proportional_error_model(create_model_for_test):
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

    model = create_model_for_test(code)
    assert has_proportional_error_model(model)

    code = """$PROBLEM base model
$INPUT ID DV TIME DVID
$DATA file.csv IGNORE=@

$PRED
IF (DVID.EQ.1) THEN
    Y = THETA(1) + ETA(1) + ERR(1)
ELSE
    Y = THETA(1) + THETA(1) * EPS(2)
END IF

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$SIGMA 1
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert not has_proportional_error_model(model)
    assert not has_proportional_error_model(model, 1)
    assert has_proportional_error_model(model, 2)


def test_has_combined_error_model(create_model_for_test):
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

    model = create_model_for_test(code)
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

    code = """$PROBLEM base model
$INPUT ID DV TIME DVID
$DATA file.csv IGNORE=@

$PRED
CONC = THETA(1)
IF (DVID.EQ.1) THEN
    Y = (CONC + ETA(1)) + (CONC + ETA(1)) * ERR(1) + ERR(2)
ELSE
    Y = CONC + ERR(3)
END IF

$THETA 0.1
$OMEGA 0.01
$SIGMA 1
$SIGMA .02
$SIGMA .02
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    model = read_model_from_string(code)
    assert has_combined_error_model(model)
    assert has_combined_error_model(model, 1)
    assert not has_combined_error_model(model, 2)


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
    model = use_thetas_for_error_stdev(model)
    correct = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
Y = THETA(1) + ETA(1) + EPS(1)*THETA(2)

$THETA 0.1
$THETA  (0,1.0) ; SD_EPS_1
$OMEGA 0.01
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""

    assert model.model_code == correct

    model = read_model_from_string(code)
    model = set_weighted_error_model(model)
    model = use_thetas_for_error_stdev(model)

    correct = """$PROBLEM base model
$INPUT ID DV TIME
$DATA file.csv IGNORE=@

$PRED
W = THETA(2)
Y = THETA(1) + ETA(1) + EPS(1)*W

$THETA 0.1
$THETA  (0,1.0) ; SD_EPS_1
$OMEGA 0.01
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVALS=9990 PRINT=2 POSTHOC
"""
    assert model.model_code == correct

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
    model = set_weighted_error_model(model)
    model = use_thetas_for_error_stdev(model)

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
W = CONC*THETA(3)
Y = CONC + EPS(1)*W
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,0.11506954418958998) ; SD_EPS_1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTERACTION
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
    model = set_weighted_error_model(model)
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
    assert model.model_code == correct

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
    model = set_weighted_error_model(model)
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


def test_has_weighted_error_model():
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
    assert not has_weighted_error_model(model)
    model = set_weighted_error_model(model)
    assert has_weighted_error_model(model)

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
    assert not has_weighted_error_model(model)
    model = set_weighted_error_model(model)
    assert has_weighted_error_model(model)


def test_set_dtbs_error_model(tmp_path, create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
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
    model = create_model_for_test(code, dataset='pheno')
    model = model.replace(name='run1')
    model = set_dtbs_error_model(model)

    with chdir(tmp_path):
        model.write_files()
        with open('run1_contr.f90') as fh:
            assert fh.readline().startswith('      subroutine contr')
        with open('run1_ccontra.f90') as fh:
            assert fh.readline().startswith('      subroutine ccontr')

    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
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
$THETA  (0,0.11506954418958998) ; SD_EPS_1
$THETA  (0,0.11506954418958998) ; SD_EPS_2
$THETA  1 ; tbs_lambda
$THETA  0.001 ; tbs_zeta
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_time_varying_error_model():
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
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
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    model = set_time_varying_error_model(model, cutoff=1.0)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$ERROR
IF (TIME.LT.1.0) THEN
    Y = F + EPS(1)*F*THETA(3)
ELSE
    Y = F + EPS(1)*F
END IF
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  0.1 ; time_varying
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_time_varying_error_model_multiple_dvs(testdata, load_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$ERROR
IF (DVID.EQ.1) THEN
    Y = F + F*EPS(1)
ELSE
    Y = F + F*EPS(1) + EPS(2)
END IF
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = read_model_from_string(code)
    model = set_time_varying_error_model(model, cutoff=1.0, dv=1)

    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$ERROR
IF (TIME.LT.1.0) THEN
    Y_1 = F + EPS(1)*F*THETA(3)
ELSE
    Y_1 = F + EPS(1)*F
END IF
Y_2 = F + EPS(1)*F + EPS(2)
IF (DVID.EQ.1) THEN
    Y = Y_1
ELSE
    Y = Y_2
END IF
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  0.1 ; time_varying
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


@pytest.mark.parametrize(
    'epsilons, same_eta, eta_names, err_ref, omega_ref',
    [
        (
            ['EPS_1'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            ['EPS_1', 'EPS_2'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n'
            'IPRED = F + EPS(2)*EXP(ETA_RV2)\n'
            'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2',
        ),
        (
            ['EPS_1', 'EPS_3'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n'
            'IPRED=F+EPS(2)\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA_RV2)\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2',
        ),
        (
            None,
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n'
            'IPRED = F + EPS(2)*EXP(ETA_RV2)\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA_RV3)\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2\n' '$OMEGA  0.09 ; IIV_RUV3',
        ),
        (
            None,
            True,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n'
            'IPRED = F + EPS(2)*EXP(ETA_RV1)\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA_RV1)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            ['EPS_1'],
            False,
            ['ETA_3'],
            'Y = F + EPS(1)*W*EXP(ETA(3))\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            'EPS_1',
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA_RV1)\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
    ],
)
def test_set_iiv_on_ruv(
    create_model_for_test,
    load_model_for_test,
    pheno_path,
    epsilons,
    same_eta,
    eta_names,
    err_ref,
    omega_ref,
):
    model = load_model_for_test(pheno_path)

    model_str = model.model_code
    model_more_eps = re.sub(
        'IPRED=F\nIRES=DV-IPRED', 'IPRED=F+EPS(2)\nIRES=DV-IPRED+EPS(3)', model_str
    )
    model_sigma = re.sub(
        r'\$SIGMA 0.013241', '$SIGMA 0.013241\n$SIGMA 0.1\n$SIGMA 0.1', model_more_eps
    )
    model = create_model_for_test(model_sigma)

    model = set_iiv_on_ruv(model, list_of_eps=epsilons, same_eta=same_eta, eta_names=eta_names)

    assert eta_names is None or eta_names[0] in model.random_variables.etas.names

    err_rec = model.internals.control_stream.get_records('ERROR')[0]

    assert str(err_rec) == f'$ERROR\n' f'W=F\n' f'{err_ref}' f'IWRES=IRES/W\n\n'

    omega_rec = ''.join(str(rec) for rec in model.internals.control_stream.get_records('OMEGA'))

    assert omega_rec == (
        f'$OMEGA DIAGONAL(2)\n'
        f' 0.0309626  ;       IVCL\n'
        f' 0.031128  ;        IVV\n\n'
        f'{omega_ref}\n'
    )


def test_set_iiv_on_ruv_multiple_dvs(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_dvid.mod')
    model = set_iiv_on_ruv(model, dv=1)
    rec = model.internals.control_stream.get_records('ERROR')[0]
    correct = """$ERROR
Y_1 = F + EPS(1)*F*EXP(ETA_RV1)
Y_2 = F + EPS(1)*F + EPS(2)

IF (DVID.EQ.1) THEN
    Y = Y_1
ELSE
    Y = Y_2
END IF
"""
    assert str(rec) == correct


@pytest.mark.parametrize(
    'epsilons, err_ref, theta_ref',
    [
        (
            ['EPS_1'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED=F+EPS(2)\n'
            'IRES=DV-IPRED+EPS(3)',
            '$THETA  (0.01,1) ; power1',
        ),
        (
            ['EPS_1', 'EPS_2'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED = F + EPS(2)*F**THETA(5)\n'
            'IRES=DV-IPRED+EPS(3)',
            '$THETA  (0.01,1) ; power1\n' '$THETA  (0.01,1) ; power2',
        ),
        (
            ['EPS_1', 'EPS_3'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED=F+EPS(2)\n'
            'IRES = DV - IPRED + EPS(3)*F**THETA(5)',
            '$THETA  (0.01,1) ; power1\n' '$THETA  (0.01,1) ; power2',
        ),
        (
            None,
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED = F + EPS(2)*F**THETA(5)\n'
            'IRES = DV - IPRED + EPS(3)*F**THETA(6)',
            '$THETA  (0.01,1) ; power1\n' '$THETA  (0.01,1) ; power2\n' '$THETA  (0.01,1) ; power3',
        ),
    ],
)
def test_set_power_on_ruv(
    load_model_for_test, create_model_for_test, testdata, epsilons, err_ref, theta_ref, tmp_path
):
    shutil.copy(testdata / 'nonmem/pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno_real.phi', tmp_path / 'run1.phi')
    shutil.copy(testdata / 'nonmem/pheno_real.ext', tmp_path / 'run1.ext')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with chdir(tmp_path):
        model_pheno = load_model_for_test('run1.mod')
        model_more_eps = re.sub(
            r'V=TVV\*EXP\(ETA\(2\)\)',
            'V=TVV',
            model_pheno.model_code,
        )
        model_more_eps = re.sub(
            r'( 0.031128  ;        IVV\n)',
            '$SIGMA 0.1\n$SIGMA 0.1',
            model_more_eps,
        )
        model_more_eps = re.sub(
            r'IPRED=F\nIRES=DV-IPRED',
            r'IPRED=F+EPS(2)\nIRES=DV-IPRED+EPS(3)',
            model_more_eps,
        )
        model = create_model_for_test(model_more_eps)
        model = model.replace(dataset=model_pheno.dataset)

        model = set_power_on_ruv(model, epsilons, zero_protection=True)

        rec_err = str(model.internals.control_stream.get_records('ERROR')[0])
        correct = f'$ERROR\n' f'W=F\n' f'{err_ref}\n' f'IWRES=IRES/W\n\n'
        assert rec_err == correct

        rec_theta = ''.join(str(rec) for rec in model.internals.control_stream.get_records('THETA'))

        assert (
            rec_theta == f'$THETA (0,0.00469307) ; PTVCL\n'
            f'$THETA (0,1.00916) ; PTVV\n'
            f'$THETA (-.99,.1)\n'
            f'{theta_ref}\n'
        )


def test_set_power_on_ruv_with_zero_protect(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = remove_error_model(model)
    model = set_proportional_error_model(model)
    model = set_power_on_ruv(model)

    assert 'F + EPS(1)*IPREDADJ**THETA(3)' in model.model_code
