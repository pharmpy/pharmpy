import sympy

from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.model import output
from pharmpy.modeling import add_peripheral_compartment, set_michaelis_menten_elimination


def test_des(load_model_for_test, create_model_for_test, testdata):
    code = """$PROBLEM
$INPUT ID TIME DV AMT
$DATA data.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=6
$MODEL COMP=(DOSE) COMP=(CENTRAL) COMP=(PERIPH) COMP=(LIVER)
$PK
MAT = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
VC = THETA(3) * EXP(ETA(3))
Q = THETA(4) * EXP(ETA(4))
VP = THETA(5) * EXP(ETA(5))
VM = THETA(6) * EXP(ETA(6))
KM = THETA(7) * EXP(ETA(7))
KLO = THETA(8) * EXP(ETA(8))
VL = THETA(9) * EXP(ETA(9))
KA = 1/MAT
$DES
DADT(1) = -(KA * A(1))
DADT(2) = KA * A(1) + Q/VP * A(3) - (Q/VC * A(2) + A(2)/VC * VM/(KM + A(2)/VC))
DADT(3) = Q/VC * A(2) - Q/VP * A(3)
DADT(4) = A(2)/VC * VM/(KM + A(2)/VC) - KLO * A(4)
$ERROR
CONC = A(2)/VC
Y = CONC + EPS(1)
$ESTIMATION METHOD=COND INTER
$COVARIANCE PRINT=E
$THETA (0, 1, Inf) ; POP_MAT
$THETA (0, 1, Inf) ; POP_CL
$THETA (0, 1, Inf) ; POP_VC
$THETA (0, 1, Inf) ; POP_Q
$THETA (0, 1, Inf) ; POP_VP
$THETA (0, 1, Inf) ; POP_VM
$THETA (0, 1, Inf) ; POP_KM
$THETA (0, 1, Inf) ; POP_KLO
$THETA (0, 1, Inf) ; POP_VL
$OMEGA 0.1; IIV_MAT
$OMEGA 0.1; IIV_CL
$OMEGA 0.1; IIV_VC
$OMEGA 0.1; IIV_Q
$OMEGA 0.1; IIV_VP
$OMEGA 0.1; IIV_VM
$OMEGA 0.1; IIV_KM
$OMEGA 0.1; IIV_KLO
$OMEGA 0.1; IIV_VL
$SIGMA 0.1; RUV_ADD
"""
    pheno = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = create_model_for_test(code)
    model = model.replace(dataset=pheno.dataset)
    cs = model.statements.ode_system
    assert len(cs) == 4


def test_conversion_round_trip(load_example_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    odes = model.statements.ode_system
    central = odes.central_compartment
    assert odes.get_flow(central, output) == parse_expr('CLMM*KM/(V1*(KM + A_CENTRAL(t)/V1))')


def test_des_mm(load_example_model_for_test, create_model_for_test):
    model = load_example_model_for_test('pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    model = set_michaelis_menten_elimination(model)
    code = model.model_code
    dataset = model.dataset
    model = create_model_for_test(code)
    model = model.replace(dataset=dataset)
    cs = model.statements.ode_system
    central = cs.central_compartment
    assert cs.get_flow(central, output) == sympy.simplify(
        parse_expr('CLMM*KM/(V1*(KM + A_CENTRAL(t)/V1))')
    )
