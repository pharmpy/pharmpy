from io import StringIO

import pytest
import sympy

from pharmpy import Model
from pharmpy.plugins.nonmem.advan import compartmental_model


def S(x):
    return sympy.Symbol(x)


@pytest.mark.parametrize(
    'advan,trans,compmat,amounts,strodes,corrics',
    [
        (
            'ADVAN1',
            'TRANS1',
            [[-S('K'), 0], [S('K'), 0]],
            [S('A_CENTRAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_CENTRAL(t), t), -K*A_CENTRAL(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {sympy.Function('A_CENTRAL')(0): S('AMT'), sympy.Function('A_OUTPUT')(0): 0},
        ),
        (
            'ADVAN1',
            'TRANS2',
            [[-S('CL') / S('V'), 0], [S('CL') / S('V'), 0]],
            [S('A_CENTRAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_CENTRAL(t), t), -CL*A_CENTRAL(t)/V)',
                'Eq(Derivative(A_OUTPUT(t), t), CL*A_CENTRAL(t)/V)',
            ],
            {sympy.Function('A_CENTRAL')(0): S('AMT'), sympy.Function('A_OUTPUT')(0): 0},
        ),
        (
            'ADVAN2',
            'TRANS1',
            [[-S('KA'), 0, 0], [S('KA'), -S('K'), 0], [0, S('K'), 0]],
            [S('A_DEPOT'), S('A_CENTRAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_DEPOT(t), t), -KA*A_DEPOT(t))',
                'Eq(Derivative(A_CENTRAL(t), t), -K*A_CENTRAL(t) + KA*A_DEPOT(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {
                sympy.Function('A_DEPOT')(0): S('AMT'),
                sympy.Function('A_CENTRAL')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
        (
            'ADVAN2',
            'TRANS2',
            [[-S('KA'), 0, 0], [S('KA'), -S('CL') / S('V'), 0], [0, S('CL') / S('V'), 0]],
            [S('A_DEPOT'), S('A_CENTRAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_DEPOT(t), t), -KA*A_DEPOT(t))',
                'Eq(Derivative(A_CENTRAL(t), t), -CL*A_CENTRAL(t)/V + KA*A_DEPOT(t))',
                'Eq(Derivative(A_OUTPUT(t), t), CL*A_CENTRAL(t)/V)',
            ],
            {
                sympy.Function('A_DEPOT')(0): S('AMT'),
                sympy.Function('A_CENTRAL')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
        (
            'ADVAN3',
            'TRANS1',
            [[-S('K12') - S('K'), S('K21'), 0], [S('K12'), -S('K21'), 0], [S('K'), 0, 0]],
            [S('A_CENTRAL'), S('A_PERIPHERAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_CENTRAL(t), t), K21*A_PERIPHERAL(t) + (-K - K12)*A_CENTRAL(t))',
                'Eq(Derivative(A_PERIPHERAL(t), t), K12*A_CENTRAL(t) - K21*A_PERIPHERAL(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {
                sympy.Function('A_CENTRAL')(0): S('AMT'),
                sympy.Function('A_PERIPHERAL')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
        (
            'ADVAN4',
            'TRANS1',
            [
                [-S('KA'), 0, 0, 0],
                [S('KA'), -S('K23') - S('K'), S('K32'), 0],
                [0, S('K23'), -S('K32'), 0],
                [0, S('K'), 0, 0],
            ],
            [S('A_DEPOT'), S('A_CENTRAL'), S('A_PERIPHERAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_DEPOT(t), t), -KA*A_DEPOT(t))',
                'Eq(Derivative(A_CENTRAL(t), t), K32*A_PERIPHERAL(t) + '
                'KA*A_DEPOT(t) + (-K - K23)*A_CENTRAL(t))',
                'Eq(Derivative(A_PERIPHERAL(t), t), K23*A_CENTRAL(t) - K32*A_PERIPHERAL(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {
                sympy.Function('A_DEPOT')(0): S('AMT'),
                sympy.Function('A_CENTRAL')(0): 0,
                sympy.Function('A_PERIPHERAL')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
        (
            'ADVAN10',
            'TRANS1',
            [
                [-S('VM') / (S('KM') + sympy.Function('A_CENTRAL')(S('t'))), 0],
                [S('VM') / (S('KM') + sympy.Function('A_CENTRAL')(S('t'))), 0],
            ],
            [S('A_CENTRAL'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_CENTRAL(t), t), -VM*A_CENTRAL(t)/(KM + A_CENTRAL(t)))',
                'Eq(Derivative(A_OUTPUT(t), t), VM*A_CENTRAL(t)/(KM + A_CENTRAL(t)))',
            ],
            {sympy.Function('A_CENTRAL')(0): S('AMT'), sympy.Function('A_OUTPUT')(0): 0},
        ),
        (
            'ADVAN11',
            'TRANS1',
            [
                [-S('K12') - S('K13') - S('K'), S('K21'), S('K31'), 0],
                [S('K12'), -S('K21'), 0, 0],
                [S('K13'), 0, -S('K31'), 0],
                [S('K'), 0, 0, 0],
            ],
            [S('A_CENTRAL'), S('A_PERIPHERAL1'), S('A_PERIPHERAL2'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_CENTRAL(t), t), K21*A_PERIPHERAL1(t) + K31*A_PERIPHERAL2(t) + '
                '(-K - K12 - K13)*A_CENTRAL(t))',
                'Eq(Derivative(A_PERIPHERAL1(t), t), K12*A_CENTRAL(t) - K21*A_PERIPHERAL1(t))',
                'Eq(Derivative(A_PERIPHERAL2(t), t), K13*A_CENTRAL(t) - K31*A_PERIPHERAL2(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {
                sympy.Function('A_CENTRAL')(0): S('AMT'),
                sympy.Function('A_PERIPHERAL1')(0): 0,
                sympy.Function('A_PERIPHERAL2')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
        (
            'ADVAN12',
            'TRANS1',
            [
                [-S('KA'), 0, 0, 0, 0],
                [S('KA'), -S('K23') - S('K24') - S('K'), S('K32'), S('K42'), 0],
                [0, S('K23'), -S('K32'), 0, 0],
                [0, S('K24'), 0, -S('K42'), 0],
                [0, S('K'), 0, 0, 0],
            ],
            [S('A_DEPOT'), S('A_CENTRAL'), S('A_PERIPHERAL1'), S('A_PERIPHERAL2'), S('A_OUTPUT')],
            [
                'Eq(Derivative(A_DEPOT(t), t), -KA*A_DEPOT(t))',
                'Eq(Derivative(A_CENTRAL(t), t), K32*A_PERIPHERAL1(t) + '
                'K42*A_PERIPHERAL2(t) + KA*A_DEPOT(t) + (-K - K23 - K24)*A_CENTRAL(t))',
                'Eq(Derivative(A_PERIPHERAL1(t), t), K23*A_CENTRAL(t) - K32*A_PERIPHERAL1(t))',
                'Eq(Derivative(A_PERIPHERAL2(t), t), K24*A_CENTRAL(t) - K42*A_PERIPHERAL2(t))',
                'Eq(Derivative(A_OUTPUT(t), t), K*A_CENTRAL(t))',
            ],
            {
                sympy.Function('A_DEPOT')(0): S('AMT'),
                sympy.Function('A_CENTRAL')(0): 0,
                sympy.Function('A_PERIPHERAL1')(0): 0,
                sympy.Function('A_PERIPHERAL2')(0): 0,
                sympy.Function('A_OUTPUT')(0): 0,
            },
        ),
    ],
)
def test_pheno(pheno, advan, trans, compmat, amounts, strodes, corrics):
    cm, ass = compartmental_model(pheno, advan, trans)

    assert ass.symbol == S('F')
    assert ass.expression == S('A_CENTRAL') / S('S1')
    assert cm.compartmental_matrix == sympy.Matrix(compmat)
    assert cm.amounts == sympy.Matrix(amounts)
    exodes = cm.to_explicit_system()
    odes, ics = exodes.odes, exodes.ics
    odes = [str(ode) for ode in odes]
    assert odes == strodes
    assert ics == corrics


def test_advan5(testdata):
    model = Model.create_model(testdata / 'nonmem' / 'DDMODEL00000130')
    cm, ass = compartmental_model(model, 'ADVAN5', 'TRANS1')
    assert ass.symbol == S('F')
    assert ass.expression == S('A_CMS1')
    assert cm.amounts == sympy.Matrix(
        [S('A_CMS1'), S('A_CMS2'), S('A_INTM'), S('A_INTM2'), S('A_COL1'), S('A_OUTPUT')]
    )
    # compmat = sympy.Matrix(
    #    [
    #        [-S('K12') - S('K10') - S('K14'), S('K21'), S('K31'), 0, 0, 0],
    #        [S('K12'), -S('K21') - S('K25'), 0, 0, 0, 0],
    #        [0, 0, -S('K31') - S('K30'), S('K43'), 0, 0],
    #        [S('K14'), 0, 0, -S('K43') - S('K40') - S('K45'), S('K54'), 0],
    #        [0, S('K25'), 0, S('K45'), -S('K54'), 0],
    #        [S('K10'), 0, S('K30'), S('K40'), 0, 0],
    #    ]
    # )
    # FIXME: Problematic because of Pharmpy renumbering.
    # How keep original numbering AND being able to add new compartments
    # assert cm.compartmental_matrix == compmat


def test_rate_constants():
    code = """$PROBLEM
$INPUT ID VISI DAT1 DGRP DOSE FLAG ONO XIME DVO NEUY SCR AGE SEX
       NYHA WT COMP IACE DIG DIU NUMB TAD TIME VID CRCL AMT SS II
       VID1 CMT CONO DV EVID OVID SHR SHR2
$DATA mx19B.csv IGNORE=@
$SUBROUTINE ADVAN5 TRANS1
$MODEL COMPARTMENT=(TRANSIT1 DEFDOSE) COMPARTMENT=(TRANSIT2) COMPARTMENT=(TRANSIT3)
COMPARTMENT=(TRANSIT4) COMPARTMENT=(TRANSIT5) COMPARTMENT=(TRANSIT6)
COMPARTMENT=(TRANSIT7) COMPARTMENT=(TRANSIT8) COMPARTMENT=(DEPOT) COMPARTMENT=(CENTRAL)
$PK
MDT = THETA(4)*EXP(ETA(4))
CL = THETA(1) * EXP(ETA(1))
VC = THETA(2) * EXP(ETA(2))
MAT = THETA(3) * EXP(ETA(3))
KA = 1/MAT
V = VC
K12 = 8/MDT
K910 = KA
K100 = CL/V
K23 = 8/MDT
K34 = 8/MDT
K45 = 8/MDT
K56 = 8/MDT
K67 = 8/MDT
K78 = 8/MDT
K89 = 8/MDT
$ERROR
CONC = A(10)/VC
Y = CONC + CONC * EPS(1)
$ESTIMATION METHOD=1 INTER MAXEVAL=9999
$COVARIANCE PRINT=E
$THETA  (0,22.7,Inf) ; POP_CL
$THETA  (0,128,Inf) ; POP_VC
$THETA  (0,0.196,Inf) ; POP_MAT
$THETA  (0,0.003) ; POP_MDT
$OMEGA  0.001  ;     IIV_CL
$OMEGA  0.001  ;     IIV_VC
$OMEGA  0.001  ;    IIV_MAT
$OMEGA  0.001 ; IIV_MDT
$SIGMA  0.273617  ;   RUV_PROP
"""
    model = Model.create_model(StringIO(code))
    odes = model.statements.ode_system
    assert odes.get_flow(odes.central_compartment, odes.output_compartment) == sympy.Symbol('K100')
