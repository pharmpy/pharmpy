from pharmpy.modeling import (
    add_metabolite,
    add_peripheral_compartment,
    has_presystemic_metabolite,
    remove_peripheral_compartment,
    transform_blq,
)


def test_add_metabolite(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
    assert not has_presystemic_metabolite(model)
    a = model.code.split('\n')
    assert a[20] == 'IF (DVID.EQ.1) Y = Y'
    assert a[21] == 'IF (DVID.EQ.2) Y = Y_M'

    assert odes.central_compartment.name == 'CENTRAL'

    model = add_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()

    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()


def test_add_metabolite_with_blq(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = transform_blq(model, method='m3', lloq=0.5)
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
    assert not has_presystemic_metabolite(model)
    a = model.code.split('\n')
    assert a[33] == 'IF (DVID.EQ.1) Y = Y'
    assert a[34] == 'IF (DVID.EQ.2) Y = Y_M'


def test_add_metabolite_transits_nodepot(create_model_for_test):
    model = create_model_for_test("""
$PROBLEM TRANSITS(10, NODEPOT);PERIPHERALS(1)
$INPUT ID TIME AMT DV DVID EVID DOSEA BW BMI HEIGHT TEMP DIABP OXYSAT PULSE RESP SYSBP Capsule AGE
$DATA ../.datasets/data2.csv IGNORE=@ IGNORE=(DVID.EQN.2)
$SUBROUTINES ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE)
COMPARTMENT=(TRANSIT2) COMPARTMENT=(TRANSIT3) COMPARTMENT=(TRANSIT4)
COMPARTMENT=(TRANSIT5) COMPARTMENT=(TRANSIT6) COMPARTMENT=(TRANSIT7)
COMPARTMENT=(TRANSIT8) COMPARTMENT=(TRANSIT9) COMPARTMENT=(TRANSIT10)
COMPARTMENT=(TRANSIT11) COMPARTMENT=(CENTRAL) COMPARTMENT=(PERIPHERAL1)
$ABBR REPLACE ETA_CL=ETA(1)
$ABBR REPLACE ETA_VC=ETA(2)
$ABBR REPLACE ETA_MAT=ETA(3)
$PK
VP1 = THETA(5)
QP1 = THETA(4)
MDT = THETA(3)*EXP(ETA_MAT)
CL = THETA(1)*EXP(ETA_CL)
VC = THETA(2)*EXP(ETA_VC)
V = VC
K12 = 11/MDT
K23 = 11/MDT
K34 = 11/MDT
K45 = 11/MDT
K56 = 11/MDT
K67 = 11/MDT
K78 = 11/MDT
K89 = 11/MDT
K9T10 = 11/MDT
K10T11 = 11/MDT
K11T12 = 11/MDT
K12T13 = QP1/V
K13T12 = QP1/VP1
K120 = CL/V
$ERROR
IPRED = A(12)/VC
IF (IPRED.EQ.0) THEN
    IPREDADJ = 0.00215000000000000
ELSE
    IPREDADJ = IPRED
END IF
Y = IPRED + EPS(1)*IPREDADJ
$THETA  (0,27278.3) ; POP_CL
$THETA  (0,438641.0) ; POP_VC
$THETA  (0,1.57195) ; POP_MDT
$THETA  (0,27278.3,999999) ; POP_QP1
$THETA  (0,21932.050000000003,999999) ; POP_VP1
$OMEGA BLOCK(2)
0.0842328 ; IIV_CL
-0.202046 ; IIV_CL_IIV_VC
0.721758 ; IIV_VC
$OMEGA  0.0340675 ; IIV_MDT
$SIGMA  0.352776 ; sigma
$TABLE ID TIME DV CIPREDI PRED CWRES FILE=mytab ONEHEADER NOAPPEND NOPRINT
$ESTIMATION METHOD=COND INTER MAXEVAL=99999
""")

    model = add_metabolite(model)

    assert model.statements.ode_system.find_compartment("METABOLITE") is not None


def test_presystemic_metabolite(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = add_metabolite(model, presystemic=True)
    odes = model.statements.ode_system

    assert odes.compartment_names == ['DEPOT', 'CENTRAL', 'METABOLITE']

    depot = odes.find_depot(model.statements)
    central = odes.central_compartment
    metabolite = odes.find_compartment("METABOLITE")

    assert has_presystemic_metabolite(model)
    assert odes.get_flow(depot, central)
    assert odes.get_flow(depot, metabolite)
    assert odes.get_flow(central, metabolite)

    assert odes.central_compartment.name == 'CENTRAL'

    model = add_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()

    model = add_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model, "METABOLITE")
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert odes.find_peripheral_compartments()

    model = remove_peripheral_compartment(model)
    odes = model.statements.ode_system

    assert not odes.find_peripheral_compartments("METABOLITE")
    assert not odes.find_peripheral_compartments()
