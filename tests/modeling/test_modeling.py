import os
import re
import shutil
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import sympy

from pharmpy import Model
from pharmpy.modeling import (
    add_covariate_effect,
    add_iiv,
    add_iov,
    add_lag_time,
    add_peripheral_compartment,
    add_pk_iiv,
    create_joint_distribution,
    fix_parameters_to,
    has_first_order_elimination,
    has_michaelis_menten_elimination,
    has_mixed_mm_fo_elimination,
    has_zero_order_absorption,
    has_zero_order_elimination,
    load_example_model,
    remove_iiv,
    remove_iov,
    remove_lag_time,
    set_bolus_absorption,
    set_first_order_absorption,
    set_first_order_elimination,
    set_iiv_on_ruv,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_ode_solver,
    set_peripheral_compartments,
    set_power_on_ruv,
    set_seq_zo_fo_absorption,
    set_transit_compartments,
    set_zero_order_absorption,
    set_zero_order_elimination,
    split_joint_distribution,
    transform_etas_boxcox,
    transform_etas_john_draper,
    transform_etas_tdist,
    update_inits,
)
from pharmpy.modeling.odes import find_clearance_parameters, find_volume_parameters
from pharmpy.utils import TemporaryDirectoryChanger


def test_set_first_order_elimination(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    correct = model.model_code
    set_first_order_elimination(model)
    assert model.model_code == correct
    assert has_first_order_elimination(model)
    set_zero_order_elimination(model)
    set_first_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CL
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    set_michaelis_menten_elimination(model)
    set_first_order_elimination(model)
    assert model.model_code == correct
    set_mixed_mm_fo_elimination(model)
    set_first_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CL
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_zero_order_elimination(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    assert not has_zero_order_elimination(model)
    set_zero_order_elimination(model)
    assert has_zero_order_elimination(model)
    assert not has_michaelis_menten_elimination(model)
    assert not has_first_order_elimination(model)
    assert not has_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CLMM
$THETA  (0,0.067,1358.0) FIX ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    set_zero_order_elimination(model)
    assert model.model_code == correct
    set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CLMM
$THETA  (0,0.067,1358.0) ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    set_mixed_mm_fo_elimination(model)
    set_zero_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CLMM = THETA(3)
KM = THETA(2)
V = THETA(1)*EXP(ETA(1))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0,135.8,1358.0) FIX ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_michaelis_menten_elimination(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    assert not has_michaelis_menten_elimination(model)
    set_michaelis_menten_elimination(model)
    assert has_michaelis_menten_elimination(model)
    assert not has_zero_order_elimination(model)
    assert not has_first_order_elimination(model)
    assert not has_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CLMM
$THETA  (0,135.8,1358.0) ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    set_michaelis_menten_elimination(model)
    assert model.model_code == correct

    set_zero_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0.0,0.00469307) ; POP_CLMM
$THETA  (0,135.8,1358.0) FIX ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_fo_mm_eta(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL = THETA(1)*EXP(ETA(1))
V = THETA(2)*EXP(ETA(2))
S1=V
$ERROR
Y=F+F*EPS(1)
$THETA  (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
$OMEGA 0.25  ; IIV_CL
$OMEGA 0.5  ; IIV_V
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(2)*EXP(ETA(1))
V = THETA(1)*EXP(ETA(2))
S1=V
$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)
$THETA (0,1.00916) ; POP_V
$THETA  (0.0,0.00469307) ; POP_CLMM
$THETA  (0,135.8,1358.0) ; POP_KM
$OMEGA  0.25 ; IIV_CLMM
$OMEGA 0.5  ; IIV_V
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_michaelis_menten_elimination_from_k(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS1
$PK
K=THETA(1)*EXP(ETA(1))
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; TVCL
$OMEGA 0.0309626  ; IVCL
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
DUMMYETA = ETA(1)
CLMM = THETA(3)
VC = THETA(2)
KM = THETA(1)
$DES
DADT(1) = -A(1)*CLMM*KM/(VC*(A(1)/VC + KM))
$ERROR
Y=F+F*EPS(1)
$THETA  (0,135.8,1358.0) ; POP_KM
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.00469307) ; POP_CLMM
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_combined_mm_fo_elimination(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
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
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    assert not has_mixed_mm_fo_elimination(model)
    set_mixed_mm_fo_elimination(model)
    assert has_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CLMM = THETA(4)
KM = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V
$DES
DADT(1) = -A(1)*(CL + CLMM*KM/(A(1)/V + KM))/V
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,135.8,1358.0) ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    set_mixed_mm_fo_elimination(model)
    assert model.model_code == correct
    set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CLMM = THETA(3)
KM = THETA(2)
V = THETA(1)*EXP(ETA(1))
S1=V
$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)
$THETA (0,1.00916) ; TVV
$THETA  (0,135.8,1358.0) ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_combined_mm_fo_elimination_from_k(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS1
$PK
K=THETA(1)*EXP(ETA(1))
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; TVCL
$OMEGA 0.0309626  ; IVCL
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
DUMMYETA = ETA(1)
CLMM = THETA(4)
VC = THETA(3)
CL = THETA(2)
KM = THETA(1)
$DES
DADT(1) = -A(1)*(CL + CLMM*KM/(A(1)/VC + KM))/VC
$ERROR
Y=F+F*EPS(1)
$THETA  (0,135.8,1358.0) ; POP_KM
$THETA  (0,0.002346535) ; POP_CL
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct

    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_zero_order_elimination(model)
    set_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
DUMMYETA = ETA(1)
CL = THETA(4)
CLMM = THETA(3)
VC = THETA(2)
KM = THETA(1)
$DES
DADT(1) = A(1)*(-CL/VC - CLMM*KM/(VC*(A(1)/VC + KM)))
$ERROR
Y=F+F*EPS(1)
$THETA  (0,0.067,1358.0) ; POP_KM
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.00469307) ; POP_CLMM
$THETA  (0,0.1) ; POP_CL
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_transit_compartments(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_transit_compartments(model, 0)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 1)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 1
    correct = (
        """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANS1 DEFDOSE) COMPARTMENT=(DEPOT) COMPARTMENT=(CENTRAL) """
        + """COMPARTMENT=(PERIPHERAL)
$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K23 = THETA(6)
K30 = CL/V
K34 = THETA(4)
K43 = THETA(5)
K12 = THETA(7)
S3 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$THETA (1,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )
    assert model.model_code == correct
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 4)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 4
    correct = (
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANS1 DEFDOSE) COMPARTMENT=(TRANS2) COMPARTMENT=(TRANSIT3) '''
        + '''COMPARTMENT=(TRANSIT4) COMPARTMENT=(DEPOT) COMPARTMENT=(CENTRAL) COMPARTMENT=(PERIPHERAL)
$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K56 = THETA(6)
K60 = CL/V
K67 = THETA(4)
K76 = THETA(5)
K12=THETA(7)
K23=THETA(7)
S6 = V
K34 = THETA(7)
K45 = THETA(7)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$THETA (1,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    )
    assert model.model_code == correct
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    set_transit_compartments(model, 1)

    assert not re.search(r'K *= *', model.model_code)
    assert re.search('K30 = CL/V', model.model_code)


def test_transits_absfo(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    set_transit_compartments(model, 0, keep_depot=False)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    assert len(model.statements.ode_system) == 2

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 1, keep_depot=False)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    assert len(model.statements.ode_system) == 4
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS1
$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K20 = CL/V
K23 = THETA(4)
K32 = THETA(5)
K12 = THETA(6)
S2 = V
KA = K12

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    assert model.model_code == correct

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    set_transit_compartments(model, 4, keep_depot=False)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 4
    correct = (
        '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANS1 DEFDOSE) COMPARTMENT=(TRANS2) COMPARTMENT=(TRANSIT3) '''
        + '''COMPARTMENT=(TRANSIT4) COMPARTMENT=(CENTRAL) COMPARTMENT=(PERIPHERAL)
$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K50 = CL/V
K56 = THETA(4)
K65 = THETA(5)
K12 = THETA(6)
K23 = THETA(6)
S5 = V
K34 = THETA(6)
K45 = THETA(6)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA (1,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    )
    assert model.model_code == correct
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    set_transit_compartments(model, 1, keep_depot=False)

    assert not re.search(r'K *= *', model.model_code)
    assert re.search('KA = 1/MDT', model.model_code)


def test_transit_compartments_added_mdt(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    set_transit_compartments(model, 2)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 2
    correct = (
        """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(CENTRAL) """
        + """COMPARTMENT=(PERIPHERAL)
$PK
MDT = THETA(6)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K30 = CL/V
K34 = THETA(4)
K43 = THETA(5)
S3 = V
K12 = 2/MDT
K23 = 2/MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$THETA  (0,0.5) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )
    assert model.model_code == correct


def test_transit_compartments_change_advan(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_transit_compartments(model, 3)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 3
    correct = (
        """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN5 TRANS1

$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(TRANSIT3) """
        + """COMPARTMENT=(CENTRAL)
$PK
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S4 = V
K12 = 3/MDT
K23 = 3/MDT
K34 = 3/MDT
K40 = CL/V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.5) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    )
    assert model.model_code == correct


def test_transit_compartments_change_number(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    set_transit_compartments(model, 3)
    set_transit_compartments(model, 2)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN5 TRANS1

$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(CENTRAL)
$PK
MDT = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S3 = V
K12 = 2/MDT
K23 = 2/MDT
K30 = CL/V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,0.5) ; POP_MDT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct

    model = Model.create_model(testdata / 'nonmem' / 'pheno.mod')
    set_transit_compartments(model, 2)
    set_transit_compartments(model, 3)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN5 TRANS1

$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(TRANSIT3) COMPARTMENT=(CENTRAL)
$PK
MDT = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S4 = V
K12 = 3/MDT
K23 = 3/MDT
K34 = 3/MDT
K40 = CL/V

$ERROR
Y=F+F*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,0.5) ; POP_MDT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""  # noqa: E501
    assert model.model_code == correct


def test_transits_non_linear_elim_with_update(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_transit_compartments(model, 3)
    model.model_code
    set_zero_order_elimination(model)
    assert 'VC1 =' not in model.model_code
    assert 'CLMM = THETA(4)*EXP(ETA(1))' in model.model_code
    assert 'CL =' not in model.model_code

    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_transit_compartments(model, 3)
    model.model_code
    set_michaelis_menten_elimination(model)
    assert 'VC1 =' not in model.model_code
    assert 'CLMM = THETA(4)*EXP(ETA(1))' in model.model_code
    assert 'CL =' not in model.model_code

    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_transit_compartments(model, 3)
    model.model_code
    set_mixed_mm_fo_elimination(model)
    assert 'VC1 =' not in model.model_code
    assert 'CLMM = THETA(6)' in model.model_code
    assert 'CL = THETA(1) * EXP(ETA(1))' in model.model_code


def test_lag_time(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    before = model.model_code
    add_lag_time(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
ALAG1 = MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.5) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct

    remove_lag_time(model)
    assert model.model_code == before


def test_add_lag_time_updated_dose(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    add_lag_time(model)
    set_first_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(5)
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
ALAG2 = MDT
KA = 1/MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.5) ; POP_MDT
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct

    set_zero_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN1 TRANS2

$PK
MAT = THETA(5)
MDT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1 = V
ALAG1 = MDT
D1 = 2*MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,0.5) ; POP_MDT
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_nested_transit_peripherals(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_transit_compartments(model, 1)
    model.model_code
    set_peripheral_compartments(model, 1)
    model.model_code
    set_peripheral_compartments(model, 2)


def test_nan_add_covariate_effect(pheno_path):
    model = Model.create_model(pheno_path)
    data = model.dataset

    new_col = [np.nan] * 10 + ([1.0] * (len(data.index) - 10))

    data['new_col'] = new_col
    model.dataset = data

    add_covariate_effect(model, 'CL', 'new_col', 'cat')
    model.update_source(nofiles=True)

    assert not re.search('NaN', model.model_code)
    assert re.search(r'NEW_COL\.EQ\.-99', model.model_code)


def test_nested_add_covariate_effect(pheno_path):
    model = Model.create_model(pheno_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')

    with pytest.warns(UserWarning):
        add_covariate_effect(model, 'CL', 'WGT', 'exp')

    model = Model.create_model(pheno_path)

    add_covariate_effect(model, 'CL', 'WGT', 'exp')
    add_covariate_effect(model, 'CL', 'APGR', 'exp')

    assert 'CL = CL*CLAPGR*CLWGT' in model.model_code
    assert 'CL = CL*CLWGT' not in model.model_code


@pytest.mark.parametrize(
    'model_path, effects, expected',
    [
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'exp', '+')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL + CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'pow', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = (WGT/WGT_MEDIAN)**THETA(4)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'lin', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'FA1', 'cat', '*')],
            '$PK\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'IF (FA1.EQ.0) THEN\n'
            '    CLFA1 = 1\n'
            'ELSE IF (FA1.EQ.1.0) THEN\n'
            '    CLFA1 = THETA(4) + 1\n'
            'END IF\n'
            'CL = CL*CLFA1\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'piece_lin', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'IF (WGT.LE.WGT_MEDIAN) THEN\n'
            '    CLWGT = THETA(4)*(WGT - WGT_MEDIAN) + 1\n'
            'ELSE\n'
            '    CLWGT = THETA(5)*(WGT - WGT_MEDIAN) + 1\n'
            'END IF\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + median', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4) - WGT + WGT_MEDIAN\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta - cov + std', '*')],
            '$PK\n'
            'WGT_STD = 0.704565\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4) - WGT + WGT_STD\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', 'theta1 * (cov/median)**theta2', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT/WGT_MEDIAN)**THETA(5)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [('CL', 'WGT', '((cov/std) - median) * theta', '*')],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'WGT_STD = 0.704565\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = THETA(4)*(WGT/WGT_STD - WGT_MEDIAN)\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '+'),
                ('V', 'WGT', 'exp', '+'),
            ],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL + CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            'V = V + VWGT\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'pheno_real.mod'),
            [
                ('CL', 'WGT', 'exp', '*'),
                ('V', 'WGT', 'exp', '*'),
            ],
            '$PK\n'
            'WGT_MEDIAN = 1.30000\n'
            'IF(AMT.GT.0) BTIME=TIME\n'
            'TAD=TIME-BTIME\n'
            'TVCL=THETA(1)*WGT\n'
            'TVV=THETA(2)*WGT\n'
            'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
            'CL=TVCL*EXP(ETA(1))\n'
            'CLWGT = EXP(THETA(4)*(WGT - WGT_MEDIAN))\n'
            'CL = CL*CLWGT\n'
            'V=TVV*EXP(ETA(2))\n'
            'VWGT = EXP(THETA(5)*(WGT - WGT_MEDIAN))\n'
            'V = V*VWGT\n'
            'S1=V\n\n',
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
            ],
            '$PK\n'
            'AGE_MEDIAN = 66.0000\n'
            'WT_MEDIAN = 78.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            'VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            'V = V*VAGE*VWT\n',
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'WT', 'exp', '*'),
                ('V', 'AGE', 'exp', '*'),
                ('V', 'CRCL', 'exp', '*'),
            ],
            '$PK\n'
            'CRCL_MEDIAN = 65.0000\n'
            'AGE_MEDIAN = 66.0000\n'
            'WT_MEDIAN = 78.0000\n'
            'CL = THETA(1) * EXP(ETA(1))\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VWT = EXP(THETA(4)*(WT - WT_MEDIAN))\n'
            'VAGE = EXP(THETA(5)*(AGE - AGE_MEDIAN))\n'
            'VCRCL = EXP(THETA(6)*(CRCL - CRCL_MEDIAN))\n'
            'V = V*VAGE*VCRCL*VWT\n',
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('CL', 'CRCL', 'exp', '*'),
                ('V', 'CRCL', 'exp', '*'),
            ],
            '$PK\n'
            'CRCL_MEDIAN = 65.0000\n'
            'CL = THETA(1)*EXP(ETA(1))\n'
            'CLCRCL = EXP(THETA(4)*(CRCL - CRCL_MEDIAN))\n'
            'CL = CL*CLCRCL\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VCRCL = EXP(THETA(5)*(CRCL - CRCL_MEDIAN))\n'
            'V = V*VCRCL\n',
        ),
        (
            ('nonmem', 'models', 'mox2.mod'),
            [
                ('V', 'CRCL', 'exp', '*'),
                ('CL', 'CRCL', 'exp', '*'),
            ],
            '$PK\n'
            'CRCL_MEDIAN = 65.0000\n'
            'CL = THETA(1)*EXP(ETA(1))\n'
            'CLCRCL = EXP(THETA(5)*(CRCL - CRCL_MEDIAN))\n'
            'CL = CL*CLCRCL\n'
            'VC = THETA(2) * EXP(ETA(2))\n'
            'MAT = THETA(3) * EXP(ETA(3))\n'
            'KA = 1/MAT\n'
            'V = VC\n'
            'VCRCL = EXP(THETA(4)*(CRCL - CRCL_MEDIAN))\n'
            'V = V*VCRCL\n',
        ),
    ],
    ids=repr,
)
def test_add_covariate_effect(testdata, model_path, effects, expected):
    model = Model.create_model(testdata.joinpath(*model_path))
    error_record_before = ''.join(map(str, model.control_stream.get_records('ERROR')))

    for effect in effects:
        add_covariate_effect(model, *effect)

    model.update_source()
    error_record_after = ''.join(map(str, model.control_stream.get_records('ERROR')))

    assert str(model.get_pred_pk_record()) == expected
    assert error_record_after == error_record_before

    for effect in effects:
        assert f'POP_{effect[0]}{effect[1]}' in model.model_code


def test_add_depot(testdata):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))

$ERROR
CONC = A(1)/V
Y = CONC + CONC*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_first_order_absorption(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
KA = 1/MAT

$ERROR
CONC = A(2)/V
Y = CONC + CONC*EPS(1)

$THETA (0,0.00469307) ; TVCL
$THETA (0,1.00916) ; TVV
$THETA  (0,2.0) ; POP_MAT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_absrate_from_explicit():
    code = """
$PROBLEM    PHENOBARB SIMPLE MODEL
$DATA      pheno.dta IGNORE=@
$INPUT      ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))

$DES
DADT(1) = -A(1)*CL/V

$ERROR
CONC = A(1)/V
Y = CONC + CONC*EPS(1)

$THETA (0,0.00469307) ; pCL
$THETA  (0,1.00916) ; pV
$OMEGA  DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
$SIGMA  1e-7
$ESTIMATION METHOD=1 INTERACTION
"""
    model = Model.create_model(StringIO(code))
    model.dataset = load_example_model("pheno").dataset
    set_first_order_absorption(model)
    correct = """
$PROBLEM    PHENOBARB SIMPLE MODEL
$DATA      pheno.dta IGNORE=@
$INPUT      ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(DEPOT DEFDOSE) COMPARTMENT=(CENTRAL)
$PK
MAT = THETA(3)
CL=THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))

$DES
DADT(1) = -A(1)*KA
DADT(2) = A(1)*KA - A(2)*CL/V

$ERROR
CONC = A(2)/V
Y = CONC + CONC*EPS(1)

$THETA (0,0.00469307) ; pCL
$THETA  (0,1.00916) ; pV
$THETA  (0,2.0) ; POP_MAT
$OMEGA  DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV
$SIGMA  1e-7
$ESTIMATION METHOD=1 INTERACTION
"""
    # FIXME: This requires more work in update
    assert correct
    # assert model.model_code == correct


def test_absorption_rate(testdata, tmp_path):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    advan1_before = model.model_code
    set_bolus_absorption(model)
    assert advan1_before == model.model_code

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    set_bolus_absorption(model)
    assert model.model_code == advan1_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    advan3_before = model.model_code
    set_bolus_absorption(model)
    assert model.model_code == advan3_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan4.mod')
    set_bolus_absorption(model)
    assert model.model_code == advan3_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan11.mod')
    advan11_before = model.model_code
    set_bolus_absorption(model)
    assert model.model_code == advan11_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan12.mod')
    set_bolus_absorption(model)
    assert model.model_code == advan11_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    advan5_nodepot_before = model.model_code
    set_bolus_absorption(model)
    assert model.model_code == advan5_nodepot_before

    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    set_bolus_absorption(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA ../pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS1
$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
K = CL/V
K12 = THETA(4)
K21 = THETA(5)
S1 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,10)
$THETA (0,10)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
"""
    assert model.model_code == correct

    # 0-order to 0-order
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    advan1_zero_order_before = model.model_code
    set_zero_order_absorption(model)
    assert model.model_code == advan1_zero_order_before

    # 0-order to Bolus
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    set_bolus_absorption(model)
    model.update_source(nofiles=True)
    assert model.model_code.split('\n')[2:] == advan1_before.split('\n')[2:]

    # 1st order to 1st order
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    advan2_before = model.model_code
    set_first_order_absorption(model)
    model.update_source(nofiles=True)
    assert model.model_code == advan2_before

    # 0-order to 1st order
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    set_first_order_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan1_zero_order.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
KA = 1/MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct

    # Bolus to 1st order
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_first_order_absorption(model)
    model.update_source(nofiles=True)
    assert model.model_code.split('\n')[2:] == correct.split('\n')[2:]

    # Bolus to 0-order
    datadir = testdata / 'nonmem' / 'modeling'
    (tmp_path / 'abs').mkdir()
    shutil.copy(datadir / 'pheno_advan1.mod', tmp_path / 'abs')
    shutil.copy(datadir / 'pheno_advan2.mod', tmp_path / 'abs')
    shutil.copy(datadir.parent / 'pheno.dta', tmp_path)
    model = Model.create_model(tmp_path / 'abs' / 'pheno_advan1.mod')
    set_zero_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN1 TRANS2

$PK
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
D1 = 2*MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct

    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan2.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN1 TRANS2

$PK
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
D1 = 2*MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''

    # 1st to 0-order
    model = Model.create_model(tmp_path / 'abs' / 'pheno_advan2.mod')
    set_zero_order_absorption(model)
    model.update_source(nofiles=True)
    assert model.model_code == correct


def test_seq_to_FO(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    set_first_order_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan2_seq.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2=V
KA = THETA(4)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_seq_to_ZO(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    set_zero_order_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT RATE WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2

$PK
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1 = V
D1 = THETA(4)

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1)
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_bolus_to_seq(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    set_seq_zo_fo_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN2 TRANS2

$PK
MDT = THETA(5)
MAT = THETA(4)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
KA = 1/MAT
D1 = 2*MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA  (0,2.0) ; POP_MAT
$THETA  (0,0.5) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_ZO_to_seq(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    set_seq_zo_fo_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_zero_order.csv IGNORE=@
$INPUT ID TIME AMT RATE WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2

$PK
MAT = THETA(5)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S2 = V
D1 = THETA(4)
KA = 1/MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1)
$THETA  (0,2.0) ; POP_MAT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_FO_to_seq(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    set_seq_zo_fo_absorption(model)
    model.update_source(nofiles=True)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno_advan2.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2 RATE
$SUBROUTINE ADVAN2 TRANS2

$PK
MDT = THETA(5)
IF(AMT.GT.0) BTIME=TIME
TAD=TIME-BTIME
TVCL=THETA(1)*WGT
TVV=THETA(2)*WGT
IF(APGR.LT.5) TVV=TVV*(1+THETA(3))
CL=TVCL*EXP(ETA(1))
V=TVV*EXP(ETA(2))
S1=V
KA=THETA(4)
D1 = 2*MDT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; CL
$THETA (0,1.00916) ; V
$THETA (-.99,.1)
$THETA (0,0.1) ; KA
$THETA  (0,0.5) ; POP_MDT
$OMEGA DIAGONAL(2)
 0.0309626  ;       IVCL
 0.031128  ;        IVV

$SIGMA 1e-7
$ESTIMATION METHOD=1 INTERACTION
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV AMT WGT APGR IPRED PRED RES TAD CWRES NPDE NOAPPEND
       NOPRINT ONEHEADER FILE=sdtab1
'''
    assert model.model_code == correct


def test_absorption_keep_mat(testdata):
    # FO to ZO (start model with MAT-eta)
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_zero_order_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.model_code
    assert 'KA =' not in model.model_code
    assert 'D1 =' in model.model_code

    # FO to seq-ZO-FO
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_seq_zo_fo_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.model_code
    assert 'KA =' in model.model_code
    assert 'D1 =' in model.model_code

    # ZO to seq-ZO-FO
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_zero_order_absorption(model)
    set_seq_zo_fo_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.model_code
    assert 'KA =' in model.model_code
    assert 'D1 =' in model.model_code
    assert 'MAT1' not in model.model_code

    # ZO to FO
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_zero_order_absorption(model)
    set_first_order_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.model_code
    assert 'KA =' in model.model_code
    assert 'D1 =' not in model.model_code

    # Transit without keeping depot
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_transit_compartments(model, 3, keep_depot=False)
    assert 'MDT = THETA(3)*EXP(ETA(3))' in model.model_code


def test_has_zero_order_absorption(pheno_path):
    model = Model.create_model(pheno_path)
    assert not has_zero_order_absorption(model)
    set_zero_order_absorption(model)
    assert has_zero_order_absorption(model)


def test_lag_on_nl_elim(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_zero_order_elimination(model)
    model.model_code
    add_lag_time(model)
    assert 'ALAG' in model.model_code


def test_zo_abs_on_nl_elim(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'models' / 'mox2.mod')
    set_zero_order_elimination(model)
    model.model_code
    set_zero_order_absorption(model)
    assert 'RATE' in model.model_code
    assert 'D1 =' in model.model_code


@pytest.mark.parametrize(
    'etas, etab, buf_new',
    [
        (
            ['ETA(1)'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
        (
            ['ETA(1)', 'ETA(2)'],
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
            'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
            'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)',
        ),
        (
            None,
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)\n'
            'ETAB2 = (EXP(ETA(2))**THETA(5) - 1)/THETA(5)',
            'CL = TVCL*EXP(ETAB1)\nV = TVV*EXP(ETAB2)',
        ),
        (
            'ETA(1)',
            'ETAB1 = (EXP(ETA(1))**THETA(4) - 1)/THETA(4)',
            'CL = TVCL*EXP(ETAB1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_transform_etas_boxcox(pheno_path, etas, etab, buf_new):
    model = Model.create_model(pheno_path)

    transform_etas_boxcox(model, etas)
    model.update_source()

    rec_ref = (
        f'$PK\n'
        f'{etab}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'{buf_new}\n'
        f'S1=V\n\n'
    )

    assert str(model.get_pred_pk_record()) == rec_ref
    assert model.parameters['lambda1'].init == 0.01


def test_transform_etas_tdist(pheno_path):
    model = Model.create_model(pheno_path)

    transform_etas_tdist(model, ['ETA(1)'])
    model.update_source()

    symbol = 'ETAT1'

    eta = 'ETA(1)'
    theta = 'THETA(4)'

    num_1 = f'{eta}**2 + 1'
    denom_1 = f'4*{theta}'

    num_2 = f'5*{eta}**4 + 16*{eta}**2 + 3'
    denom_2 = f'96*{theta}**2'

    num_3 = f'3*{eta}**6 + 19*{eta}**4 + 17*{eta}**2 - 15'
    denom_3 = f'384*{theta}**3'

    expression = (
        f'ETA(1)*(1 + ({num_1})/({denom_1}) + ({num_2})/({denom_2}) + ' f'({num_3})/({denom_3}))'
    )

    rec_ref = (
        f'$PK\n'
        f'{symbol} = {expression}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL = TVCL*EXP(ETAT1)\n'
        f'V=TVV*EXP(ETA(2))\n'
        f'S1=V\n\n'
    )

    assert str(model.get_pred_pk_record()) == rec_ref
    assert model.parameters['df1'].init == 80


@pytest.mark.parametrize(
    'etas, etad, buf_new',
    [
        (
            ['ETA(1)'],
            'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(ETA(1)*THETA(4))',
            'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))',
        ),
        (
            'ETA(1)',
            'ETAD1 = ((ABS(ETA(1)) + 1)**THETA(4) - 1)*ABS(ETA(1))/(ETA(1)*THETA(4))',
            'CL = TVCL*EXP(ETAD1)\nV=TVV*EXP(ETA(2))',
        ),
    ],
)
def test_transform_etas_john_draper(pheno_path, etas, etad, buf_new):
    model = Model.create_model(pheno_path)

    transform_etas_john_draper(model, etas)
    model.update_source()

    rec_ref = (
        f'$PK\n'
        f'{etad}\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'{buf_new}\n'
        f'S1=V\n\n'
    )

    assert str(model.get_pred_pk_record()) == rec_ref


@pytest.mark.parametrize(
    'parameter, expression, operation, eta_name, buf_new, no_of_omega_recs',
    [
        ('S1', 'exp', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + EXP(ETA(3))', 2),
        ('S1', 'exp', '*', None, 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA(3))', 2),
        ('V', 'exp', '+', None, 'V = TVV*EXP(ETA(2)) + EXP(ETA(3))\nS1=V', 2),
        ('S1', 'add', None, None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA(3)', 2),
        ('S1', 'prop', None, None, 'V=TVV*EXP(ETA(2))\nS1 = ETA(3)*V', 2),
        ('S1', 'log', None, None, 'V=TVV*EXP(ETA(2))\nS1 = V*EXP(ETA(3))/(EXP(ETA(3)) + 1)', 2),
        ('S1', 'eta_new', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA(3)', 2),
        ('S1', 'eta_new**2', '+', None, 'V=TVV*EXP(ETA(2))\nS1 = V + ETA(3)**2', 2),
        ('S1', 'exp', '+', 'ETA(3)', 'V=TVV*EXP(ETA(2))\nS1 = V + EXP(ETA(3))', 2),
        (
            ['V', 'S1'],
            'exp',
            '+',
            None,
            'V = TVV*EXP(ETA(2)) + EXP(ETA(3))\nS1 = V + EXP(ETA(4))',
            3,
        ),
        (
            ['V', 'S1'],
            'exp',
            '+',
            ['new_eta1', 'new_eta2'],
            'V = TVV*EXP(ETA(2)) + EXP(ETA(3))\nS1 = V + EXP(ETA(4))',
            3,
        ),
    ],
)
def test_add_iiv(pheno_path, parameter, expression, operation, eta_name, buf_new, no_of_omega_recs):
    model = Model.create_model(pheno_path)

    add_iiv(
        model,
        list_of_parameters=parameter,
        expression=expression,
        operation=operation,
        eta_names=eta_name,
    )

    etas = [eta.name for eta in model.random_variables.etas]

    assert eta_name is None or set(eta_name).intersection(etas) or eta_name in etas

    model.update_source()

    rec_ref = (
        f'$PK\n'
        f'IF(AMT.GT.0) BTIME=TIME\n'
        f'TAD=TIME-BTIME\n'
        f'TVCL=THETA(1)*WGT\n'
        f'TVV=THETA(2)*WGT\n'
        f'IF(APGR.LT.5) TVV=TVV*(1+THETA(3))\n'
        f'CL=TVCL*EXP(ETA(1))\n'
        f'{buf_new}\n\n'
    )

    assert str(model.get_pred_pk_record()) == rec_ref

    omega_rec = model.control_stream.get_records('OMEGA')

    assert len(omega_rec) == no_of_omega_recs
    assert '$OMEGA  0.09 ; IIV_' in str(omega_rec[-1])


def test_add_iiv_missing_param(pheno_path):
    model = Model.create_model(pheno_path)
    with pytest.raises(ValueError):
        add_iiv(model, 'non_existing_param', 'add')


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            ['ETA(1)', 'ETA(2)'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(3)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(3))\n'
            'S1 = V + ETA(2)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.1\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA(2)', 'ETA(3)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(2)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(2)', 'ETA(4)'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1 = V + ETA(4)\n'
            'MAT = THETA(3)*EXP(ETA(3))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(3)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0309626\n'
            '$OMEGA  0.1\n'
            '$OMEGA  0.031128\n',
        ),
        (
            ['ETA(2)', 'ETA(3)', 'ETA(4)'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(3)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '$OMEGA  0.031128\n',
        ),
        (
            ['ETA(3)', 'ETA(4)', 'ETA(5)'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA DIAGONAL(2)\n'
            '0.0309626  ; IVCL\n'
            '0.031128  ; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            None,
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(5)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031045\t; IIV_CL_IIV_Q\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_create_joint_distribution(testdata, etas, pk_ref, omega_ref):
    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')

    create_joint_distribution(model, etas)
    model.update_source()
    assert str(model.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            (['ETA(1)', 'ETA(2)'], ['ETA(1)', 'ETA(3)']),
            '$PK\nCL=THETA(1)*EXP(ETA(1))\nV = THETA(2)*EXP(ETA(3))\n'
            'S1 = V + ETA(2)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.1\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            (None, ['ETA(1)', 'ETA(2)']),
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(2)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            (['ETA(1)', 'ETA(2)'], None),
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA BLOCK(5)\n'
            '0.0309626\t; IVCL\n'
            '0.0031045\t; IIV_CL_IIV_V\n'
            '0.031128\t; IVV\n'
            '0.0055644\t; IIV_CL_IIV_S1\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0030963\t; IIV_CL_IIV_MAT\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031045\t; IIV_CL_IIV_Q\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_create_joint_distribution_nested(testdata, etas, pk_ref, omega_ref):
    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')

    create_joint_distribution(model, etas[0])
    model.update_source()
    create_joint_distribution(model, etas[1])
    model.update_source()

    assert str(model.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            ['ETA(1)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(4)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(2)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA BLOCK(3)\n'
            '0.1\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(3)'],
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V = THETA(2)*EXP(ETA(3))\n'
            'S1 = V + ETA(2)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.1\n'
            '$OMEGA BLOCK(3)\n'
            '0.031128\t; IVV\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
        (
            None,
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA  0.1\n'
            '$OMEGA  0.0309626\n'
            '$OMEGA  0.031128\n',
        ),
        (
            'ETA(1)',
            '$PK\nCL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n\n',
            '$OMEGA  0.0309626 ; IVCL\n'
            '$OMEGA BLOCK(4)\n'
            '0.031128\t; IVV\n'
            '0.0055792\t; IIV_V_IIV_S1\n'
            '0.1\n'
            '0.0031045\t; IIV_V_IIV_MAT\n'
            '0.0055644\t; IIV_S1_IIV_MAT\n'
            '0.0309626\n'
            '0.0031128\t; IIV_V_IIV_Q\n'
            '0.0055792\t; IIV_S1_IIV_Q\n'
            '0.0005\n'
            '0.031128\n',
        ),
    ],
)
def test_split_joint_distribution(testdata, etas, pk_ref, omega_ref):
    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')
    create_joint_distribution(model)
    model.update_source()

    split_joint_distribution(model, etas)
    model.update_source()

    assert str(model.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


@pytest.mark.parametrize(
    'epsilons, same_eta, eta_names, err_ref, omega_ref',
    [
        (
            ['EPS(1)'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            ['EPS(1)', 'EPS(2)'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n'
            'IPRED = F + EPS(2)*EXP(ETA(4))\n'
            'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2',
        ),
        (
            ['EPS(1)', 'EPS(3)'],
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n'
            'IPRED = F + EPS(2)\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA(4))\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2',
        ),
        (
            None,
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n'
            'IPRED = F + EPS(2)*EXP(ETA(4))\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA(5))\n',
            '$OMEGA  0.09 ; IIV_RUV1\n' '$OMEGA  0.09 ; IIV_RUV2\n' '$OMEGA  0.09 ; IIV_RUV3',
        ),
        (
            None,
            True,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n'
            'IPRED = F + EPS(2)*EXP(ETA(3))\n'
            'IRES = DV - IPRED + EPS(3)*EXP(ETA(3))\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            ['EPS(1)'],
            False,
            ['ETA(3)'],
            'Y = F + EPS(1)*W*EXP(ETA(3))\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
        (
            'EPS(1)',
            False,
            None,
            'Y = F + EPS(1)*W*EXP(ETA(3))\n' 'IPRED=F+EPS(2)\n' 'IRES=DV-IPRED+EPS(3)\n',
            '$OMEGA  0.09 ; IIV_RUV1',
        ),
    ],
)
def test_set_iiv_on_ruv(pheno_path, epsilons, same_eta, eta_names, err_ref, omega_ref):
    model = Model.create_model(pheno_path)

    model_str = model.model_code
    model_more_eps = re.sub(
        'IPRED=F\nIRES=DV-IPRED', 'IPRED=F+EPS(2)\nIRES=DV-IPRED+EPS(3)', model_str
    )
    model_sigma = re.sub(
        r'\$SIGMA 0.013241', '$SIGMA 0.013241\n$SIGMA 0.1\n$SIGMA 0.1', model_more_eps
    )
    model = Model.create_model(StringIO(model_sigma))

    print("START")
    print(model.parameters)
    set_iiv_on_ruv(model, epsilons, same_eta, eta_names)
    print(model.parameters)
    model.update_source()

    assert eta_names is None or eta_names[0] in [eta.name for eta in model.random_variables.etas]

    err_rec = model.control_stream.get_records('ERROR')[0]

    assert str(err_rec) == f'$ERROR\n' f'W=F\n' f'{err_ref}' f'IWRES=IRES/W\n\n'

    omega_rec = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert omega_rec == (
        f'$OMEGA DIAGONAL(2)\n'
        f' 0.0309626  ;       IVCL\n'
        f' 0.031128  ;        IVV\n\n'
        f'{omega_ref}\n'
    )


@pytest.mark.parametrize(
    'etas, pk_ref, omega_ref',
    [
        (
            ['ETA(1)'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)*EXP(ETA(1))\n'
            'S1 = V + ETA(2)\n'
            'MAT = THETA(3)*EXP(ETA(3))\n'
            'Q = THETA(4)*EXP(ETA(4))\n\n',
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(2)'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)\n'
            'S1 = V + ETA(1)\n'
            'MAT = THETA(3)*EXP(ETA(2))\n'
            'Q = THETA(4)*EXP(ETA(3))\n\n',
            '$OMEGA 0.1\n' '$OMEGA BLOCK(2)\n' '0.0309626\n' '0.0005 0.031128\n',
        ),
        (
            ['ETA(1)', 'ETA(4)'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)*EXP(ETA(1))\n'
            'S1 = V + ETA(2)\n'
            'MAT = THETA(3)\n'
            'Q = THETA(4)*EXP(ETA(3))\n\n',
            '$OMEGA  0.031128 ; IVV\n' '$OMEGA 0.1\n' '$OMEGA  0.031128\n',
        ),
        (
            ['ETA(4)', 'ETA(5)'],
            '$PK\n'
            'CL=THETA(1)*EXP(ETA(1))\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT = THETA(3)\n'
            'Q = THETA(4)\n\n',
            '$OMEGA DIAGONAL(2)\n' '0.0309626  ; IVCL\n' '0.031128  ; IVV\n' '$OMEGA 0.1\n',
        ),
        (
            None,
            '$PK\n'
            'DUMMYETA = ETA(1)\n'
            'CL = THETA(1)\n'
            'V = THETA(2)\n'
            'S1 = V\n'
            'MAT = THETA(3)\n'
            'Q = THETA(4)\n\n',
            '$OMEGA  0 FIX ; DUMMYOMEGA\n',
        ),
        (
            ['CL'],
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)*EXP(ETA(1))\n'
            'S1 = V + ETA(2)\n'
            'MAT = THETA(3)*EXP(ETA(3))\n'
            'Q = THETA(4)*EXP(ETA(4))\n\n',
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
        (
            'ETA(1)',
            '$PK\n'
            'CL = THETA(1)\n'
            'V = THETA(2)*EXP(ETA(1))\n'
            'S1 = V + ETA(2)\n'
            'MAT = THETA(3)*EXP(ETA(3))\n'
            'Q = THETA(4)*EXP(ETA(4))\n\n',
            '$OMEGA  0.031128 ; IVV\n'
            '$OMEGA 0.1\n'
            '$OMEGA BLOCK(2)\n'
            '0.0309626\n'
            '0.0005 0.031128\n',
        ),
    ],
)
def test_remove_iiv(testdata, etas, pk_ref, omega_ref):
    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')
    remove_iiv(model, etas)
    model.update_source()

    assert str(model.get_pred_pk_record()) == pk_ref

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == omega_ref


def test_remove_iov(testdata):
    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')

    model_str = model.model_code
    model_with_iov = model_str.replace(
        '$OMEGA DIAGONAL(2)\n' '0.0309626  ; IVCL\n' '0.031128  ; IVV',
        '$OMEGA BLOCK(1)\n0.1\n$OMEGA BLOCK(1) SAME\n',
    )

    model = Model.create_model(StringIO(model_with_iov))

    remove_iov(model)
    model.update_source()

    assert (
        str(model.get_pred_pk_record()) == '$PK\n'
        'CL = THETA(1)\n'
        'V = THETA(2)\n'
        'S1 = V + ETA(1)\n'
        'MAT = THETA(3)*EXP(ETA(2))\n'
        'Q = THETA(4)*EXP(ETA(3))\n\n'
    )
    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega == '$OMEGA 0.1\n' '$OMEGA BLOCK(2)\n' '0.0309626\n' '0.0005 0.031128\n'

    model = Model.create_model(testdata / 'nonmem/pheno_block.mod')

    with pytest.warns(UserWarning):
        remove_iov(model)


def test_remove_iov_diagonal():
    model = Model.create_model(
        StringIO(
            '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS1
$PK
K=THETA(1)*EXP(ETA(1))+ETA(2)+ETA(3)+ETA(4)+ETA(5)+ETA(6)+ETA(7)
$ERROR
Y=F+F*EPS(1)
$THETA 0.1
$OMEGA DIAGONAL(2)
0.015
0.02
$OMEGA BLOCK(1)
0.6
$OMEGA BLOCK(1) SAME
$OMEGA 0.1
$OMEGA BLOCK(1)
0.01
$OMEGA BLOCK(1) SAME
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
'''
        )
    )

    remove_iov(model)

    assert (
        '''$OMEGA DIAGONAL(2)
0.015
0.02
$OMEGA  0.1'''
        in model.model_code
    )


@pytest.mark.parametrize(
    ('distribution', 'occ', 'to_remove', 'cases', 'rest'),
    (
        (
            'disjoint',
            'VISI',
            None,
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_3 = 0\n',
            (),
        ),
        (
            'joint',
            'VISI',
            None,
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_3 = 0\n',
            (),
        ),
        (
            'disjoint',
            'VISI',
            'ETA_IOV_1_1',
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_2 = ETA(5)\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(6)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(7)\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
        (
            'joint',
            'VISI',
            'ETA_IOV_1_1',
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_2 = ETA(6)\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(5)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(7)\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
        (
            'disjoint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_2 = ETA(5)\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(6)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(7)\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
        (
            'joint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3) IOV_2 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_2 = ETA(6)\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(5)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(7)\n',
            ('ETA_IOV_2_1', 'ETA_IOV_2_2', 'ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
        (
            'disjoint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2', 'ETA_IOV_2_1'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(5)\n',
            ('ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
        (
            'joint',
            'VISI',
            ['ETA_IOV_1_1', 'ETA_IOV_1_2', 'ETA_IOV_2_1'],
            'IOV_1 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_1 = 0\n'
            'IOV_2 = 0\n'
            'IF (VISI.EQ.3.OR.VISI.EQ.8) IOV_2 = 0\n'
            'IOV_3 = 0\n'
            'IF (VISI.EQ.3) IOV_3 = ETA(4)\n'
            'IF (VISI.EQ.8) IOV_3 = ETA(5)\n',
            ('ETA_IOV_3_1', 'ETA_IOV_3_2'),
        ),
    ),
    ids=repr,
)
def test_remove_iov_with_options(tmp_path, testdata, distribution, occ, to_remove, cases, rest):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model = Model.create_model('mox2.mod')
        model.datainfo.path = tmp_path / 'mox_simulated_normal.csv'

        start_model = add_iov(model, occ=occ, distribution=distribution)
        model_with_some_iovs_removed = start_model.copy()

        remove_iov(model_with_some_iovs_removed, to_remove=to_remove)
        assert cases in model_with_some_iovs_removed.model_code
        assert set(
            map(lambda rv: rv.name, model_with_some_iovs_removed.random_variables.iov)
        ) == set(rest)


@pytest.mark.parametrize(
    'etas_file, force, file_exists',
    [('', False, False), ('', True, True), ('$ETAS FILE=run1.phi', False, True)],
)
def test_update_inits(testdata, etas_file, force, file_exists, tmp_path):
    shutil.copy(testdata / 'nonmem/pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno.phi', tmp_path / 'run1.phi')
    shutil.copy(testdata / 'nonmem/pheno.ext', tmp_path / 'run1.ext')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with TemporaryDirectoryChanger(tmp_path):
        with open('run1.mod', 'a') as f:
            f.write(etas_file)

        model = Model.create_model('run1.mod')
        update_inits(model, force)
        model.update_source()

        assert ('$ETAS FILE=run1_input.phi' in model.model_code) is file_exists
        assert (os.path.isfile('run1_input.phi')) is file_exists


def test_update_inits_move_est(pheno_path):
    model = Model.create_model(pheno_path)
    res = model.modelfit_results

    create_joint_distribution(model)
    add_iiv(model, 'S1', 'add')

    param_est = res.parameter_estimates
    param_est['IIV_CL_IIV_V'] = 0.0285  # Correlation > 0.99
    param_est['IIV_S1'] = 0.0005
    res.parameter_estimates = param_est

    update_inits(model, move_est_close_to_bounds=True)

    assert model.parameters['OMEGA(1,1)'].init == res.parameter_estimates['OMEGA(1,1)']
    assert model.parameters['IIV_S1'].init == 0.01
    assert round(model.parameters['IIV_CL_IIV_V'].init, 6) == 0.025757


def test_update_inits_zero_fix(pheno_path):
    model = Model.create_model(pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    fix_parameters_to(model, d)
    res = model.modelfit_results
    param_est = res.parameter_estimates
    del param_est['OMEGA(1,1)']
    update_inits(model)
    assert model.parameters['OMEGA(1,1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].fix

    model = Model.create_model(pheno_path)
    d = {name: 0 for name in model.random_variables.iiv.parameter_names}
    fix_parameters_to(model, d)
    res = model.modelfit_results
    param_est = res.parameter_estimates
    del param_est['OMEGA(1,1)']
    update_inits(model, move_est_close_to_bounds=True)
    assert model.parameters['OMEGA(1,1)'].init == 0
    assert model.parameters['OMEGA(1,1)'].fix


def test_update_inits_no_res(testdata, tmp_path):
    shutil.copy(testdata / 'nonmem/pheno.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with TemporaryDirectoryChanger(tmp_path):
        model = Model.create_model('run1.mod')
        with pytest.raises(ValueError):
            update_inits(model)

        shutil.copy(testdata / 'nonmem/pheno.ext', tmp_path / 'run1.ext')
        shutil.copy(testdata / 'nonmem/pheno.lst', tmp_path / 'run1.lst')

        model = Model.create_model('run1.mod')

        model.modelfit_results.parameter_estimates = pd.Series(
            np.nan, name='estimates', index=list(model.parameters.nonfixed.inits.keys())
        )

        with pytest.raises(ValueError):
            update_inits(model)


@pytest.mark.parametrize(
    'epsilons, err_ref, theta_ref',
    [
        (
            ['EPS(1)'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED=F+EPS(2)\n'
            'IRES=DV-IPRED+EPS(3)',
            '$THETA  (0.01,1) ; power1',
        ),
        (
            ['EPS(1)', 'EPS(2)'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED = F + EPS(2)*F**THETA(5)\n'
            'IRES=DV-IPRED+EPS(3)',
            '$THETA  (0.01,1) ; power1\n' '$THETA  (0.01,1) ; power2',
        ),
        (
            ['EPS(1)', 'EPS(3)'],
            'IF (F.EQ.0) F = 2.22500000000000E-307\n'
            'Y = F + EPS(1)*F**THETA(4)\n'
            'IPRED = F + EPS(2)\n'  # FIXME: registers as different despite not being changed
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
def test_set_power_on_ruv(testdata, epsilons, err_ref, theta_ref, tmp_path):
    shutil.copy(testdata / 'nonmem/pheno_real.mod', tmp_path / 'run1.mod')
    shutil.copy(testdata / 'nonmem/pheno_real.phi', tmp_path / 'run1.phi')
    shutil.copy(testdata / 'nonmem/pheno_real.ext', tmp_path / 'run1.ext')
    shutil.copy(testdata / 'nonmem/pheno.dta', tmp_path / 'pheno.dta')

    with TemporaryDirectoryChanger(tmp_path):
        model_pheno = Model.create_model('run1.mod')
        model_more_eps = re.sub(
            r'( 0.031128  ;        IVV\n)',
            '$SIGMA 0.1\n$SIGMA 0.1',
            model_pheno.model_code,
        )
        model_more_eps = re.sub(
            r'IPRED=F\nIRES=DV-IPRED',
            r'IPRED=F+EPS(2)\nIRES=DV-IPRED+EPS(3)',
            model_more_eps,
        )
        model = Model.create_model(StringIO(model_more_eps))
        model.dataset = model_pheno.dataset

        set_power_on_ruv(model, epsilons, zero_protection=True)
        model.update_source()

        rec_err = str(model.control_stream.get_records('ERROR')[0])
        correct = f'$ERROR\n' f'W=F\n' f'{err_ref}\n' f'IWRES=IRES/W\n\n'
        assert rec_err == correct

        rec_theta = ''.join(str(rec) for rec in model.control_stream.get_records('THETA'))

        assert (
            rec_theta == f'$THETA (0,0.00469307) ; PTVCL\n'
            f'$THETA (0,1.00916) ; PTVV\n'
            f'$THETA (-.99,.1)\n'
            f'{theta_ref}\n'
        )


def test_nested_update_source(pheno_path):
    model = Model.create_model(pheno_path)

    create_joint_distribution(model)
    model.update_source()
    model.update_source()

    assert 'IIV_CL_IIV_V' in model.model_code

    model = Model.create_model(pheno_path)

    remove_iiv(model, 'CL')

    model.update_source()
    model.update_source()

    assert '0.031128 ; IVV' in model.model_code
    assert '0.0309626  ;       IVCL' not in model.model_code

    model = Model.create_model(pheno_path)

    remove_iiv(model, 'V')

    model.update_source()
    model.update_source()

    assert '0.0309626 ; IVCL' in model.model_code
    assert '0.031128  ;        IVV' not in model.model_code


@pytest.mark.parametrize(
    'path, occ, etas, eta_names, pk_start_ref, pk_end_ref, omega_ref, distribution',
    [
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA(1)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            'ETA(1)',
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            'ETA(1)',
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA(1)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA(1)', 'CL'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            [['CL', 'ETA(1)']],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'explicit',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            [['ETA(1)', 'CL']],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'explicit',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(5)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(6)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_2\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(5)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(4)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(6)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_1\n'
            '0.001\t; OMEGA_IOV_1_2\n'
            '0.0031128\t; OMEGA_IOV_2\n'
            '$OMEGA BLOCK(2) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(5)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(6)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_2\n',
            'same-as-iiv',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA(2)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(2)\n',
            'CL=TVCL*EXP(ETA(1))\n' 'V = TVV*EXP(ETAI1)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'same-as-iiv',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA(2)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(5)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(6)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_2\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['CL', 'ETA(2)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(5)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(4)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(6)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V = TVV*EXP(ETAI2)\n' 'S1=V\n',
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_1\n'
            '0.001\t; OMEGA_IOV_1_2\n'
            '0.0031128\t; OMEGA_IOV_2\n'
            '$OMEGA BLOCK(2) SAME\n',
            'joint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA(1)'],
            ['ETA(3)', 'ETA(4)'],
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_real.mod',
            'FA1',
            ['ETA(1)'],
            ['ETA(3)', 'ETA(4)'],
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(3)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(4)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = TVCL*EXP(ETAI1)\n' 'V=TVV*EXP(ETA(2))\n' 'S1=V\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'joint',
        ),
        (
            'nonmem/pheno_block.mod',
            'FA1',
            ['ETA(1)'],
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(6)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(7)\n'
            'ETAI1 = IOV_1 + ETA(1)\n',
            'CL = THETA(1)*EXP(ETAI1)\n'
            'V=THETA(2)*EXP(ETA(2))\n'
            'S1=V+ETA(3)\n'
            'MAT=THETA(3)*EXP(ETA(4))\n'
            'Q=THETA(4)*EXP(ETA(5))\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n',
            'disjoint',
        ),
        (
            'nonmem/pheno_block.mod',
            'FA1',
            None,
            None,
            'IOV_1 = 0\n'
            'IF (FA1.EQ.0) IOV_1 = ETA(6)\n'
            'IF (FA1.EQ.1) IOV_1 = ETA(7)\n'
            'IOV_2 = 0\n'
            'IF (FA1.EQ.0) IOV_2 = ETA(8)\n'
            'IF (FA1.EQ.1) IOV_2 = ETA(9)\n'
            'IOV_3 = 0\n'
            'IF (FA1.EQ.0) IOV_3 = ETA(10)\n'
            'IF (FA1.EQ.1) IOV_3 = ETA(11)\n'
            'IOV_4 = 0\n'
            'IF (FA1.EQ.0) IOV_4 = ETA(12)\n'
            'IF (FA1.EQ.1) IOV_4 = ETA(14)\n'
            'IOV_5 = 0\n'
            'IF (FA1.EQ.0) IOV_5 = ETA(13)\n'
            'IF (FA1.EQ.1) IOV_5 = ETA(15)\n'
            'ETAI1 = IOV_1 + ETA(1)\n'
            'ETAI2 = IOV_2 + ETA(2)\n'
            'ETAI3 = IOV_3 + ETA(3)\n'
            'ETAI4 = IOV_4 + ETA(4)\n'
            'ETAI5 = IOV_5 + ETA(5)\n',
            'CL = THETA(1)*EXP(ETAI1)\n'
            'V = THETA(2)*EXP(ETAI2)\n'
            'S1 = ETAI3 + V\n'
            'MAT = THETA(3)*EXP(ETAI4)\n'
            'Q = THETA(4)*EXP(ETAI5)\n',
            '$OMEGA  BLOCK(1)\n'
            '0.00309626 ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_1\n'
            '$OMEGA  BLOCK(1)\n'
            '0.0031128 ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_2\n'
            '$OMEGA  BLOCK(1)\n'
            '0.010000000000000002 ; OMEGA_IOV_3\n'
            '$OMEGA  BLOCK(1) SAME ; OMEGA_IOV_3\n'
            '$OMEGA BLOCK(2)\n'
            '0.00309626\t; OMEGA_IOV_4\n'
            '5E-05\t; OMEGA_IOV_4_5\n'
            '0.0031128\t; OMEGA_IOV_5\n'
            '$OMEGA BLOCK(2) SAME\n',
            'same-as-iiv',
        ),
    ],
)
def test_add_iov(
    testdata, path, occ, etas, eta_names, pk_start_ref, pk_end_ref, omega_ref, distribution
):
    model = Model.create_model(testdata / path)
    add_iov(model, occ, etas, eta_names, distribution=distribution)
    model.update_source()

    model_etas = set(eta.name for eta in model.random_variables.etas)
    assert eta_names is None or model_etas.issuperset(eta_names)

    pk_rec = str(model.get_pred_pk_record())

    expected_pk_rec_start = f'$PK\n{pk_start_ref}'
    expected_pk_rec_end = f'{pk_end_ref}\n'

    assert pk_rec[: len(expected_pk_rec_start)] == expected_pk_rec_start
    assert pk_rec[-len(expected_pk_rec_end) :] == expected_pk_rec_end

    rec_omega = ''.join(str(rec) for rec in model.control_stream.get_records('OMEGA'))

    assert rec_omega[-len(omega_ref) :] == omega_ref


def test_add_iov_compose(pheno_path):
    model1 = Model.create_model(pheno_path)
    add_iov(model1, 'FA1', ['ETA(1)', 'ETA(2)'])
    model1.update_source()

    model2 = Model.create_model(pheno_path)
    add_iov(model2, 'FA1', 'ETA(1)')
    model2.update_source()
    add_iov(model2, 'FA1', 'ETA(2)')
    model2.update_source()

    assert set(eta.name for eta in model1.random_variables.etas) == set(
        eta.name for eta in model2.random_variables.etas
    )
    # FIXME find better way to assert models are equivalent
    assert sorted(str(model1.get_pred_pk_record()).split('\n')) == sorted(
        str(model2.get_pred_pk_record()).split('\n')
    )

    rec_omega_1 = list(str(rec) for rec in model1.control_stream.get_records('OMEGA'))
    rec_omega_2 = list(str(rec) for rec in model2.control_stream.get_records('OMEGA'))

    assert rec_omega_1 == rec_omega_2


def test_add_iov_only_one_level(pheno_path):
    model = Model.create_model(pheno_path)
    model.dataset['FA1'] = 1

    with pytest.raises(ValueError, match='Only one value in FA1 column.'):
        add_iov(model, 'FA1', ['ETA(1)'])


@pytest.mark.parametrize(
    'occ, params, new_eta_names, distribution, error, message',
    (
        (
            'FA1',
            ['ETA(1)', 'CL'],
            None,
            'disjoint',
            ValueError,
            'ETA(1) ~ N(0, OMEGA(1,1)) was given twice.',
        ),
        (
            'FA1',
            ['CL', 'ETA(1)'],
            None,
            'disjoint',
            ValueError,
            'ETA(1) ~ N(0, OMEGA(1,1)) was given twice.',
        ),
        (
            'FA1',
            [['ETA(1)'], ['CL']],
            None,
            'explicit',
            ValueError,
            'ETA(1) ~ N(0, OMEGA(1,1)) was given twice.',
        ),
        (
            'FA1',
            [['CL'], ['ETA(1)']],
            None,
            'explicit',
            ValueError,
            'ETA(1) ~ N(0, OMEGA(1,1)) was given twice.',
        ),
        (
            'FA1',
            ['ETA(1)'],
            None,
            'abracadabra',
            ValueError,
            '"abracadabra" is not a valid value for distribution',
        ),
        (
            'FA1',
            ['ETA(1)'],
            None,
            'explicit',
            ValueError,
            'distribution == "explicit" requires parameters to be given as lists of lists',
        ),
        (
            'FA1',
            [['ETA(2)'], 'ETA(1)'],
            None,
            'explicit',
            ValueError,
            'distribution == "explicit" requires parameters to be given as lists of lists',
        ),
        (
            'FA1',
            [['ETA(1)']],
            None,
            'joint',
            ValueError,
            'distribution != "explicit" requires parameters to be given as lists of strings',
        ),
        (
            'FA1',
            [['ETA(1)'], [2, 'ETA(2)']],
            None,
            'explicit',
            ValueError,
            'not all parameters are string',
        ),
        (
            'FA1',
            [['ETA(1)', 'ETA(2)']],
            ['A', 'B', 'C', 'D', 'E'],
            'explicit',
            ValueError,
            'Number of given eta names is incorrect, need 4 names.',
        ),
    ),
)
def test_add_iov_raises(pheno_path, occ, params, new_eta_names, distribution, error, message):
    model = Model.create_model(pheno_path)
    with pytest.raises(error, match=re.escape(message)):
        add_iov(model, occ, params, eta_names=new_eta_names, distribution=distribution)


def test_set_ode_solver(pheno_path):
    model = Model.create_model(pheno_path)
    assert model.estimation_steps[0].solver is None
    assert 'ADVAN1' in model.model_code

    model = Model.create_model(pheno_path)
    set_michaelis_menten_elimination(model)
    set_ode_solver(model, 'LSODA')
    assert model.estimation_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.model_code

    model = Model.create_model(pheno_path)
    set_zero_order_elimination(model)
    assert 'ADVAN13' in model.model_code
    set_ode_solver(model, 'LSODA')
    set_michaelis_menten_elimination(model)
    assert model.estimation_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.model_code
    set_ode_solver(model, 'DVERK')
    assert model.estimation_steps[0].solver == 'DVERK'
    assert 'ADVAN6' in model.model_code


def test_add_pk_iiv(pheno_path):
    model = Model.create_model(pheno_path)
    set_zero_order_elimination(model)
    add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA(1)', 'ETA(2)', 'ETA_KM'}
    add_peripheral_compartment(model)
    add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA(1)', 'ETA(2)', 'ETA_KM', 'ETA_VP1', 'ETA_QP1'}

    model = Model.create_model(pheno_path)
    set_zero_order_elimination(model)
    add_peripheral_compartment(model)
    add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA(1)', 'ETA(2)', 'ETA_KM', 'ETA_VP1', 'ETA_QP1'}


def test_add_pk_iiv_nested_params(testdata, pheno_path):
    model = Model.create_model(pheno_path)
    set_transit_compartments(model, 3)
    add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA(1)', 'ETA(2)', 'ETA_MDT'}

    model = Model.create_model(pheno_path)
    set_first_order_absorption(model)
    add_pk_iiv(model)
    iivs = set(model.random_variables.iiv.names)
    assert iivs == {'ETA(1)', 'ETA(2)', 'ETA_MAT'}

    model = Model.create_model(pheno_path)
    set_transit_compartments(model, 3)
    add_pk_iiv(model, initial_estimate=0.01)
    assert model.parameters['IIV_MDT'].init == 0.01


def test_mm_then_periph(pheno):
    model = pheno.copy()
    set_michaelis_menten_elimination(model)
    add_peripheral_compartment(model)
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.peripheral_compartments[0]
    assert odes.get_flow(central, periph) == sympy.Symbol('QP1') / sympy.Symbol('V')
    assert odes.get_flow(periph, central) == sympy.Symbol('QP1') / sympy.Symbol('VP1')
    add_peripheral_compartment(model)
    newperiph = odes.peripheral_compartments[1]
    assert odes.get_flow(central, newperiph) == sympy.Symbol('QP2') / sympy.Symbol('V')
    assert odes.get_flow(newperiph, central) == sympy.Symbol('QP2') / sympy.Symbol('VP2')


def test_find_clearance_parameters(pheno):
    model = pheno.copy()
    cl_origin = find_clearance_parameters(model)
    assert cl_origin == [sympy.Symbol('CL')]
    add_peripheral_compartment(model)
    cl_p1 = find_clearance_parameters(model)
    model.update_source()
    assert cl_p1 == [sympy.Symbol('CL'), sympy.Symbol('QP1')]
    add_peripheral_compartment(model)
    cl_p2 = find_clearance_parameters(model)
    model.update_source()
    assert cl_p2 == [sympy.Symbol('CL'), sympy.Symbol('QP1'), sympy.Symbol('QP2')]


def test_find_volume_parameters(pheno):
    model = pheno.copy()
    v_origin = find_volume_parameters(model)
    assert v_origin == [sympy.Symbol('V')]
    add_peripheral_compartment(model)
    model.update_source()
    v_p1 = find_volume_parameters(model)
    assert v_p1 == [sympy.Symbol('V1'), sympy.Symbol('VP1')]
    add_peripheral_compartment(model)
    model.update_source()
    v_p2 = find_volume_parameters(model)
    assert v_p2 == [sympy.Symbol('V1'), sympy.Symbol('VP1'), sympy.Symbol('VP2')]
