import re
import shutil
from typing import Iterable

import pytest

from pharmpy.basic import Expr, Matrix
from pharmpy.model import Assignment, Bolus, Infusion
from pharmpy.modeling import (
    add_bioavailability,
    add_lag_time,
    add_peripheral_compartment,
    create_basic_pk_model,
    find_clearance_parameters,
    find_volume_parameters,
    get_central_volume_and_clearance,
    get_initial_conditions,
    get_lag_times,
    get_zero_order_inputs,
    has_first_order_elimination,
    has_linear_odes,
    has_linear_odes_with_real_eigenvalues,
    has_michaelis_menten_elimination,
    has_mixed_mm_fo_elimination,
    has_odes,
    has_weibull_absorption,
    has_zero_order_absorption,
    has_zero_order_elimination,
    remove_bioavailability,
    remove_lag_time,
    remove_peripheral_compartment,
    set_first_order_absorption,
    set_first_order_elimination,
    set_initial_condition,
    set_instantaneous_absorption,
    set_michaelis_menten_elimination,
    set_mixed_mm_fo_elimination,
    set_ode_solver,
    set_peripheral_compartments,
    set_seq_zo_fo_absorption,
    set_tmdd,
    set_transit_compartments,
    set_weibull_absorption,
    set_zero_order_absorption,
    set_zero_order_elimination,
    set_zero_order_input,
    unload_dataset,
)
from pharmpy.modeling.odes import CompartmentalSystem, CompartmentalSystemBuilder


def test_advan1(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.find_peripheral_compartments()[0]
    rate = model.statements.ode_system.get_flow(central, periph)
    assert rate == Expr.symbol('Q') / Expr.symbol('V1')


def test_advan2(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V=VC
KA=1/MAT
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.22) ; POP_MAT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.22) ; POP_MAT
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan2_trans1(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS1
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
KA=1/MAT
K=CL/VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.22) ; POP_MAT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS1
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
KA=1/MAT
K23 = QP1/VC
K32 = QP1/VP1
K=CL/VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.22) ; POP_MAT
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan3(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.010000000000000002) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.09000000000000001) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan4(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN12 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q3 = QP1
V3 = VP1
Q4 = QP2
V4 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.010000000000000002) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.09000000000000001) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan1_two_periphs(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00046930699999999997) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$THETA  (0,0.004223763) ; POP_QP2
$THETA  (0,0.011000000000000001) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan1_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code)
    model = remove_peripheral_compartment(model)
    assert model.code == code


def test_advan3_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,2.350801373088405) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan4_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V = VC
KA=1/MAT
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,2.350801373088405) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan11_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan12_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN12 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q3 = QP1
V3 = VP1
Q4 = QP2
V4 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan11_remove_two_periphs(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,4.48160274617681) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_advan4_roundtrip(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.05) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_set_peripheral_compartments(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = set_peripheral_compartments(model, 0)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,4.48160274617681) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_set_first_order_elimination(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    correct = model.code
    model = set_first_order_elimination(model)
    assert model.code == correct
    assert has_first_order_elimination(model)
    model = set_zero_order_elimination(model)
    model = set_first_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; TVV
$OMEGA  0.0309626 ; IIV_CL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    model = set_michaelis_menten_elimination(model)
    model = set_first_order_elimination(model)
    assert model.code == correct
    model = set_mixed_mm_fo_elimination(model)
    model = set_first_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN1 TRANS2

$PK
CL = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; TVV
$OMEGA  0.0309626 ; IIV_CL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_set_zero_order_elimination(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert not has_zero_order_elimination(model)
    model = set_zero_order_elimination(model)
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
CLMM = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CLMM
$THETA (0,1.00916) ; TVV
$THETA  (0,0.067,101.85000000000001) FIX ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    model = set_zero_order_elimination(model)
    assert model.code == correct
    model = set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CLMM
$THETA (0,1.00916) ; TVV
$THETA  (0,0.067,101.85000000000001) ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_mixed_mm_fo_elimination(model)
    model = set_zero_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$ABBR REPLACE ETA_2=ETA(1)
$PK
CLMM = THETA(3)
KM = THETA(2)
V = THETA(1)*EXP(ETA_2)
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA (0,1.00916) ; TVV
$THETA  (0,33.95,101.85000000000001) FIX ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_set_michaelis_menten_elimination(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    assert not has_michaelis_menten_elimination(model)
    model = set_michaelis_menten_elimination(model)
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
CLMM = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CLMM
$THETA (0,1.00916) ; TVV
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    model = set_michaelis_menten_elimination(model)
    assert model.code == correct

    model = set_zero_order_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV
$SUBROUTINE ADVAN13 TOL=9

$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(1)*EXP(ETA(1))
V=THETA(2)*EXP(ETA(2))
S1=V

$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)

$THETA  (0,0.00469307) ; POP_CLMM
$THETA (0,1.00916) ; TVV
$THETA  (0,33.95,101.85000000000001) FIX ; POP_KM
$OMEGA  0.0309626 ; IIV_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241

$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_mm_elimination_no_data(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = unload_dataset(model)
    set_michaelis_menten_elimination(model)


def test_fo_mm_eta(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    model = set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
KM = THETA(3)
CLMM = THETA(1)*EXP(ETA(1))
V = THETA(2)*EXP(ETA(2))
S1=V
$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)
$THETA  (0,0.00469307) ; POP_CLMM
$THETA (0,1.00916) ; POP_V
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$OMEGA  0.25 ; IIV_CLMM
$OMEGA 0.5  ; IIV_V
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_set_michaelis_menten_elimination_from_k(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    model = set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
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
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.00469307) ; POP_CLMM
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_combined_mm_fo_elimination(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    assert not has_mixed_mm_fo_elimination(model)
    model = set_mixed_mm_fo_elimination(model)
    assert has_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
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
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct
    model = set_mixed_mm_fo_elimination(model)
    assert model.code == correct
    model = set_michaelis_menten_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$ABBR REPLACE ETA_2=ETA(1)
$PK
CLMM = THETA(3)
KM = THETA(2)
V = THETA(1)*EXP(ETA_2)
S1=V
$DES
DADT(1) = -A(1)*CLMM*KM/(V*(A(1)/V + KM))
$ERROR
Y=F+F*EPS(1)
$THETA (0,1.00916) ; TVV
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA 0.031128  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_combined_mm_fo_elimination_from_k(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    model = set_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
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
$THETA  (0,33.95,101.85000000000001) ; POP_KM
$THETA  (0,0.002346535) ; POP_CL
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.002346535) ; POP_CLMM
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct

    model = create_model_for_test(code, dataset='pheno')
    model = set_zero_order_elimination(model)
    model = set_mixed_mm_fo_elimination(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN13 TOL=9
$MODEL COMPARTMENT=(CENTRAL DEFDOSE)
$PK
CL = THETA(4)
DUMMYETA = ETA(1)
CLMM = THETA(3)
VC = THETA(2)
KM = THETA(1)
$DES
DADT(1) = -A(1)*(CL/VC + CLMM*KM/(VC*(A(1)/VC + KM)))
$ERROR
Y=F+F*EPS(1)
$THETA  (0,0.067,101.85000000000001) ; POP_KM
$THETA  (0,0.1) ; POP_VC
$THETA  (0,0.00469307) ; POP_CLMM
$THETA  (0,0.1) ; POP_CL
$OMEGA  0 FIX ; DUMMYOMEGA
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.code == correct


def test_transit_compartments(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = set_transit_compartments(model, 0)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    model = set_transit_compartments(model, 1)
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
K12=THETA(7)
S3 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    model = set_transit_compartments(model, 4)
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
K34 = THETA(7)
K45 = THETA(7)
K60 = CL/V
K67 = THETA(4)
K76 = THETA(5)
K12=THETA(7)
K23=THETA(7)
S6 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    model = set_transit_compartments(model, 1)

    assert not re.search(r'K *= *', model.code)
    assert re.search('K30 = CL/V', model.code)


def test_transits_absfo(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    model = set_transit_compartments(model, 0, keep_depot=False)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    assert len(model.statements.ode_system) == 1

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    model = set_transit_compartments(model, 1, keep_depot=False)
    transits = model.statements.ode_system.find_transit_compartments(model.statements)
    assert len(transits) == 0
    assert len(model.statements.ode_system) == 3
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_2transits.mod')
    model = set_transit_compartments(model, 4, keep_depot=False)
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
K34 = THETA(6)
K45 = THETA(6)
K50 = CL/V
K56 = THETA(4)
K65 = THETA(5)
K12 = THETA(6)
K23 = THETA(6)
S5 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    with pytest.raises(ValueError):
        model = set_transit_compartments(model, 1, keep_depot=False)


def test_transit_compartments_added_mdt(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    model = set_transit_compartments(model, 2)
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
K12 = 2/MDT
K23 = 2/MDT
K30 = CL/V
K34 = THETA(4)
K43 = THETA(5)
S3 = V

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct


def test_transit_compartments_change_advan(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = set_transit_compartments(model, 3)
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct


def test_transit_compartments_change_number(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_transit_compartments(model, 3)
    model = set_transit_compartments(model, 2)
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
    assert model.code == correct

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = set_transit_compartments(model, 2)
    model = set_transit_compartments(model, 3)
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
    assert model.code == correct

    model = set_transit_compartments(model, 0)


def test_transits_non_linear_elim_with_update(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_transit_compartments(model, 3)
    model = set_zero_order_elimination(model)
    assert 'VC1 =' not in model.code
    assert 'CLMM = THETA(1)*EXP(ETA(1))' in model.code
    assert 'CL =' not in model.code

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_transit_compartments(model, 3)
    model = set_michaelis_menten_elimination(model)
    assert 'VC1 =' not in model.code
    assert 'CLMM = THETA(1)*EXP(ETA(1))' in model.code
    assert 'CL =' not in model.code

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_transit_compartments(model, 3)
    model = set_mixed_mm_fo_elimination(model)
    assert 'VC1 =' not in model.code
    assert 'CLMM = THETA(6)' in model.code
    assert 'CL = THETA(1) * EXP(ETA(1))' in model.code


def test_transits_mat_to_mdt():
    model = create_basic_pk_model(administration='oral')
    model = set_transit_compartments(model, n=4, keep_depot=False)
    assert not model.statements.find_assignment('MAT')
    mdt_assign = model.statements.find_assignment('MDT')
    assert model.statements.count(mdt_assign) == 1

    model = create_basic_pk_model(administration='oral')
    model = set_transit_compartments(model, 1, keep_depot=True)
    assert model.statements.find_assignment('MAT')
    assert model.statements.find_assignment('MDT')
    model = set_transit_compartments(model, 1, keep_depot=False)
    assert not model.statements.find_assignment('MAT')
    assert model.statements.find_assignment('MDT')


def test_bioavailability(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    before = model.code
    model = add_bioavailability(model)
    assert "POP_BIO" in model.parameters
    assert model.statements.find_assignment("BIO")
    assert model.statements.find_assignment("F1")

    model = remove_bioavailability(model)
    assert model.code == before


def test_move_bioavailability(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = add_bioavailability(model)
    assert model.statements.ode_system.dosing_compartments[0].bioavailability == Expr.symbol("F1")

    model = set_first_order_absorption(model)
    assert model.statements.ode_system.dosing_compartments[0].name == "DEPOT"
    assert model.statements.ode_system.dosing_compartments[0].bioavailability == Expr.symbol("F1")
    assert not model.statements.ode_system.find_compartment("CENTRAL").doses


def test_lag_time(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    before = model.code
    model = add_lag_time(model)
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    model = remove_lag_time(model)
    assert model.code == before


def test_add_lag_time_updated_dose(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = add_lag_time(model)
    model = set_first_order_absorption(model)
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
ALAG1 = MDT
KA = 1/MAT

$ERROR
W=F
Y=F+W*EPS(1)
IPRED=F
IRES=DV-IPRED
IWRES=IRES/W

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    model = set_zero_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct


def test_nested_transit_peripherals(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_transit_compartments(model, 2)
    model = set_peripheral_compartments(model, 1)
    model = set_peripheral_compartments(model, 2)
    assert 'K64 = QP2/VP2' in model.code
    assert 'K40 = CL/V' in model.code


def test_add_depot(create_model_for_test):
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
    model = create_model_for_test(code, dataset='pheno')
    model = set_first_order_absorption(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
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
    assert model.code == correct


def test_absorption_rate(load_model_for_test, testdata, tmp_path):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    advan1_before = model.code
    model = set_instantaneous_absorption(model)
    assert advan1_before == model.code

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    model = set_instantaneous_absorption(model)
    assert model.code == advan1_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan3.mod')
    advan3_before = model.code
    model = set_instantaneous_absorption(model)
    assert model.code == advan3_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan4.mod')
    model = set_instantaneous_absorption(model)
    assert model.code == advan3_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan11.mod')
    advan11_before = model.code
    model = set_instantaneous_absorption(model)
    assert model.code == advan11_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan12.mod')
    model = set_instantaneous_absorption(model)
    assert model.code == advan11_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_nodepot.mod')
    advan5_nodepot_before = model.code
    model = set_instantaneous_absorption(model)
    assert model.code == advan5_nodepot_before

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan5_depot.mod')
    model = set_instantaneous_absorption(model)
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    # 0-order to 0-order
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    advan1_zero_order_before = model.code
    model = set_zero_order_absorption(model)
    assert model.code == advan1_zero_order_before

    # 0-order to Bolus
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    model = set_instantaneous_absorption(model)
    assert model.code.split('\n')[2:] == advan1_before.split('\n')[2:]

    # 1st order to 1st order
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    advan2_before = model.code
    model = set_first_order_absorption(model)
    assert model.code == advan2_before

    # 0-order to 1st order
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    model = set_first_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    # Bolus to 1st order
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = set_first_order_absorption(model)
    assert model.code.split('\n')[2:] == correct.split('\n')[2:]

    # Bolus to 0-order
    datadir = testdata / 'nonmem' / 'modeling'
    (tmp_path / 'abs').mkdir()
    shutil.copy(datadir / 'pheno_advan1.mod', tmp_path / 'abs')
    shutil.copy(datadir / 'pheno_advan2.mod', tmp_path / 'abs')
    shutil.copy(datadir.parent / 'pheno.dta', tmp_path)
    model = load_model_for_test(tmp_path / 'abs' / 'pheno_advan1.mod')
    model = set_zero_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct

    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    model = load_model_for_test(tmp_path / 'abs' / 'pheno_advan2.mod')
    model = set_zero_order_absorption(model)
    assert model.code == correct


def test_seq_to_FO(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    model = set_first_order_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307)
$THETA (0,1.00916)
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
    assert model.code == correct


def test_lagtime_then_zoabs(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = set_first_order_absorption(model)
    model = add_lag_time(model)
    model = set_zero_order_absorption(model)
    assert get_lag_times(model) == {'CENTRAL': Expr.symbol('ALAG1')}


def test_seq_to_ZO(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2_seq.mod')
    model = set_zero_order_absorption(model)
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

$THETA (0,0.00469307)
$THETA (0,1.00916)
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
    assert model.code == correct


def test_bolus_to_seq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    model = set_seq_zo_fo_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct


def test_ZO_to_seq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1_zero_order.mod')
    model = set_seq_zo_fo_absorption(model)
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
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
    assert model.code == correct


def test_FO_to_seq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan2.mod')
    model = set_seq_zo_fo_absorption(model)
    correct = '''$PROBLEM PHENOBARB SIMPLE MODEL
$DATA DUMMYPATH IGNORE=@
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

$THETA (0,0.00469307) ; POP_CL
$THETA (0,1.00916) ; POP_V
$THETA (-.99,.1)
$THETA (0,0.1) ; POP_KA
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
    assert model.code == correct


def test_absorption_keep_mat(load_model_for_test, testdata):
    # FO to ZO (start model with MAT-eta)
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_zero_order_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.code
    assert 'KA =' not in model.code
    assert 'D1 =' in model.code

    # FO to seq-ZO-FO
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_seq_zo_fo_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.code
    assert 'KA =' in model.code
    assert 'D1 =' in model.code

    # ZO to seq-ZO-FO
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_zero_order_absorption(model)
    model = set_seq_zo_fo_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.code
    assert 'KA =' in model.code
    assert 'D1 =' in model.code
    assert 'MAT1' not in model.code

    # ZO to FO
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_zero_order_absorption(model)
    model = set_first_order_absorption(model)
    assert 'MAT = THETA(3) * EXP(ETA(3))' in model.code
    assert 'KA =' in model.code
    assert 'D1 =' not in model.code

    # Transit without keeping depot
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_transit_compartments(model, 3, keep_depot=False)
    assert 'MDT = THETA(3)*EXP(ETA(3))' in model.code


def test_has_zero_order_absorption(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert not has_zero_order_absorption(model)
    model = set_zero_order_absorption(model)
    assert has_zero_order_absorption(model)


def test_lag_on_nl_elim(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_zero_order_elimination(model)
    model = add_lag_time(model)
    assert 'ALAG' in model.code


def test_zo_abs_on_nl_elim(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_zero_order_elimination(model)
    model = set_zero_order_absorption(model)
    assert 'RATE' in model.code
    assert 'D1 =' in model.code
    assert 'CONC = A(1)/VC' in model.code
    assert 'DADT(1) = -A(1)*' in model.code


def test_mm_then_periph(pheno):
    model = set_michaelis_menten_elimination(pheno)
    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.find_peripheral_compartments()[0]
    assert odes.get_flow(central, periph) == Expr.symbol('QP1') / Expr.symbol('V')
    assert odes.get_flow(periph, central) == Expr.symbol('QP1') / Expr.symbol('VP1')
    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system
    newperiph = odes.find_peripheral_compartments()[1]
    central = odes.central_compartment
    assert odes.get_flow(central, newperiph) == Expr.symbol('QP2') / Expr.symbol('V')
    assert odes.get_flow(newperiph, central) == Expr.symbol('QP2') / Expr.symbol('VP2')


def test_mixed_mm_fo_then_periph(pheno, load_model_for_test, testdata):
    model = set_mixed_mm_fo_elimination(pheno)
    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.find_peripheral_compartments()[0]
    assert odes.get_flow(central, periph) == Expr.symbol('QP1') / Expr.symbol('V')
    assert odes.get_flow(periph, central) == Expr.symbol('QP1') / Expr.symbol('VP1')
    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system
    newperiph = odes.find_peripheral_compartments()[1]
    central = odes.central_compartment
    assert odes.get_flow(central, newperiph) == Expr.symbol('QP2') / Expr.symbol('V')
    assert odes.get_flow(newperiph, central) == Expr.symbol('QP2') / Expr.symbol('VP2')

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = set_mixed_mm_fo_elimination(pheno)
    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.find_peripheral_compartments()[0]
    assert odes.get_flow(central, periph) == Expr.symbol('QP1') / Expr.symbol('V')
    assert odes.get_flow(periph, central) == Expr.symbol('QP1') / Expr.symbol('VP1')


def test_set_ode_solver(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    assert model.execution_steps[0].solver is None
    assert 'ADVAN1' in model.code
    assert '$MODEL' not in model.code

    model = load_model_for_test(pheno_path)
    model = set_michaelis_menten_elimination(model)
    model = set_ode_solver(model, 'LSODA')
    assert model.execution_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.code
    assert '$MODEL' in model.code

    model = load_model_for_test(pheno_path)
    model = set_zero_order_elimination(model)
    assert 'ADVAN13' in model.code
    assert '$MODEL' in model.code
    model = set_ode_solver(model, 'LSODA')
    model = set_michaelis_menten_elimination(model)
    assert model.execution_steps[0].solver == 'LSODA'
    assert 'ADVAN13' in model.code
    assert '$MODEL' in model.code
    model = set_ode_solver(model, 'DVERK')
    assert model.execution_steps[0].solver == 'DVERK'
    assert 'ADVAN6' in model.code
    assert '$MODEL' in model.code


def test_has_odes(load_example_model_for_test, datadir, load_model_for_test):
    model = load_example_model_for_test('pheno')
    assert has_odes(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_odes(model)


def test_has_linear_odes(load_example_model_for_test, datadir, load_model_for_test):
    model = load_example_model_for_test('pheno')
    assert has_linear_odes(model)
    model = set_michaelis_menten_elimination(model)
    assert not has_linear_odes(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_linear_odes(model)


def test_has_linear_odes_with_real_eigenvalues(
    load_example_model_for_test, datadir, load_model_for_test
):
    model = load_example_model_for_test('pheno')
    assert has_linear_odes_with_real_eigenvalues(model)
    model = set_michaelis_menten_elimination(model)
    assert not has_linear_odes_with_real_eigenvalues(model)
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert not has_linear_odes_with_real_eigenvalues(model)


def test_set_initial_conditions(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = set_initial_condition(model, "CENTRAL", 10)
    assert len(model.statements) == 11
    ic = Assignment.create(Expr.function('A_CENTRAL', 0), Expr.integer(10))
    assert model.statements.before_odes[-1] == ic
    assert get_initial_conditions(model) == {Expr.function('A_CENTRAL', 0): Expr.integer(10)}
    model = set_initial_condition(model, "CENTRAL", 23)
    assert len(model.statements) == 11
    ic = Assignment.create(Expr.function('A_CENTRAL', 0), Expr.integer(23))
    assert model.statements.before_odes[-1] == ic
    model = set_initial_condition(model, "CENTRAL", 0)
    assert len(model.statements) == 10


def test_get_zero_order_inputs(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    zo = get_zero_order_inputs(model)
    assert zo == Matrix([[0]])


def test_set_zero_order_input(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_zero_order_input(model, "CENTRAL", 10)
    zo = get_zero_order_inputs(model)
    assert zo == Matrix([[10]])


def test_get_initial_conditions(load_example_model_for_test, load_model_for_test, datadir):
    model = load_example_model_for_test('pheno')
    assert get_initial_conditions(model) == {Expr.function('A_CENTRAL', 0): Expr.integer(0)}
    ic = Assignment.create(Expr.function('A_CENTRAL', 0), Expr.integer(23))
    statements = (
        model.statements.before_odes
        + ic
        + model.statements.ode_system
        + model.statements.after_odes
    )
    mod2 = model.replace(statements=statements)
    assert get_initial_conditions(mod2) == {Expr.function('A_CENTRAL', 0): Expr.integer(23)}
    path = datadir / 'minimal.mod'
    model = load_model_for_test(path)
    assert get_initial_conditions(model) == {}


def _symbols(names: Iterable[str]):
    return list(map(Expr.symbol, names))


def test_find_clearance_parameters(pheno):
    cl_origin = find_clearance_parameters(pheno)
    assert cl_origin == _symbols(['CL'])

    model = add_peripheral_compartment(pheno)
    cl_p1 = find_clearance_parameters(model)
    assert cl_p1 == _symbols(['CL', 'QP1'])

    model = add_peripheral_compartment(model)
    cl_p2 = find_clearance_parameters(model)
    assert cl_p2 == _symbols(['CL', 'QP1', 'QP2'])


def test_find_clearance_parameters_github_issues_1053_and_1062(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_michaelis_menten_elimination(model)
    assert find_clearance_parameters(model) == _symbols(['CLMM'])


def test_find_clearance_parameters_github_issues_1044_and_1053(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_transit_compartments(model, 10)
    assert find_clearance_parameters(model) == _symbols(['CL'])


def test_find_clearance_parameters_github_issues_1053_and_1062_bis(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    assert find_clearance_parameters(model) == _symbols(['CLMM', 'QP1', 'QP2'])


def test_find_volume_parameters(pheno):
    v_origin = find_volume_parameters(pheno)
    assert v_origin == _symbols(['V'])

    model = add_peripheral_compartment(pheno)
    v_p1 = find_volume_parameters(model)
    assert v_p1 == _symbols(['V1', 'VP1'])

    model = add_peripheral_compartment(model)
    v_p2 = find_volume_parameters(model)
    assert v_p2 == _symbols(['V1', 'VP1', 'VP2'])


def test_find_volume_and_clearance_parameters_mm_and_mixed_mm(pheno):
    mm = set_michaelis_menten_elimination(pheno)
    assert find_volume_parameters(mm) == _symbols(['V'])
    assert find_clearance_parameters(mm) == _symbols(['CLMM'])

    mixed_mm = set_mixed_mm_fo_elimination(pheno)
    assert find_volume_parameters(mixed_mm) == _symbols(['V'])
    assert find_clearance_parameters(mixed_mm) == _symbols(['CL', 'CLMM'])


def test_find_volume_parameters_github_issues_1053_and_1062(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_michaelis_menten_elimination(model)
    assert find_volume_parameters(model) == _symbols(['VC'])


def test_find_volume_parameters_github_issues_1044_and_1053(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_transit_compartments(model, 10)
    assert find_volume_parameters(model) == _symbols(['VC'])


def test_find_volume_parameters_github_issues_1053_and_1062_bis(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    assert find_volume_parameters(model) == _symbols(['VC', 'VP1', 'VP2'])


@pytest.mark.parametrize(
    ('model_name', 'expected_cl', 'expected_v'),
    (
        ('full', ['CL'], ['VC']),
        ('ib', ['CL'], ['VC']),
        ('cr', ['CL'], ['VC']),
        ('crib', ['CL'], ['VC']),
        ('wagner', ['CL'], ['VC']),
        ('qss', ['CL'], ['VC']),
        ('mmapp', ['CL'], ['VC']),
    ),
    ids=repr,
)
def test_find_clearance_and_volume_parameters_tmdd(
    load_example_model_for_test, model_name, expected_cl, expected_v
):
    model = load_example_model_for_test('pheno')
    model = set_tmdd(model, model_name)
    assert find_clearance_parameters(model) == _symbols(expected_cl)
    assert find_volume_parameters(model) == _symbols(expected_v)

    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, 'qss')
    assert find_volume_parameters(model) == _symbols(['VC', 'VP1'])
    assert find_clearance_parameters(model) == _symbols(['CL', 'QP1'])

    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, 'wagner')
    assert find_volume_parameters(model) == _symbols(['VC', 'VP1'])
    assert find_clearance_parameters(model) == _symbols(['CL', 'QP1'])

    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = set_mixed_mm_fo_elimination(model)
    model = set_tmdd(model, 'mmapp')
    assert find_volume_parameters(model) == _symbols(['VC', 'VP1'])
    assert find_clearance_parameters(model) == _symbols(['CL', 'QP1'])


def test_multi_dose_change_absorption(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    ode = model.statements.ode_system
    cb = CompartmentalSystemBuilder(ode)
    cb.add_dose(cb.find_compartment("CENTRAL"), Bolus(Expr.symbol('AMT'), admid=2))
    ode = CompartmentalSystem(cb)
    model = model.replace(
        statements=model.statements.before_odes + ode + model.statements.after_odes
    )

    model = set_first_order_absorption(model)
    assert len(model.statements.ode_system.dosing_compartments) == 2

    depot = model.statements.ode_system.find_compartment("DEPOT")
    central = model.statements.ode_system.find_compartment("CENTRAL")

    assert depot.doses[0] == Bolus(Expr.symbol('AMT'), admid=1)
    assert central.doses[0] == Bolus(Expr.symbol('AMT'), admid=2)

    model = set_zero_order_absorption(model)
    central = model.statements.ode_system.find_compartment("CENTRAL")

    assert len(central.doses) == 2
    assert central.doses[0] == Infusion.create(
        Expr.symbol('AMT'), admid=1, duration=Expr.symbol("D1")
    )


def test_get_central_volume_and_clearance(
    testdata, load_example_model_for_test, load_model_for_test
):
    model = load_example_model_for_test("pheno")
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CL"))

    model = set_michaelis_menten_elimination(model)
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CLMM"))

    model = set_mixed_mm_fo_elimination(model)
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CL"))

    model = load_model_for_test(testdata / 'nonmem' / 'modeling' / 'pheno_advan1.mod')
    assert get_central_volume_and_clearance(model) == (Expr.symbol("V"), Expr.symbol("CL"))

    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    assert get_central_volume_and_clearance(model) == (Expr.symbol("V"), Expr.symbol("CL"))

    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CL"))

    model = set_michaelis_menten_elimination(model)
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CLMM"))

    model = add_peripheral_compartment(model)
    assert get_central_volume_and_clearance(model) == (Expr.symbol("VC"), Expr.symbol("CLMM"))


def test_issue_2161(testdata, load_model_for_test, load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_transit_compartments(model, 7, keep_depot=True)
    model = add_peripheral_compartment(model)
    assert 'K80' in model.code


def test_set_weibull_absorption(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = set_weibull_absorption(model)
    assert Expr.symbol('POP_BETA') in model.statements.before_odes.free_symbols
    assert Expr.symbol('POP_ALPHA') in model.statements.before_odes.free_symbols


def test_has_weibull_absorption(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    assert not has_weibull_absorption(model)
    model = set_weibull_absorption(model)
    assert has_weibull_absorption(model)
