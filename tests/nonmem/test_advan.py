import pytest
import sympy

import pharmpy.symbols
from pharmpy import Model
from pharmpy.plugins.nonmem.advan import compartmental_model


def S(x):
    return pharmpy.symbols.symbol(x)


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
def test_pheno(pheno_path, advan, trans, compmat, amounts, strodes, corrics):
    model = Model(pheno_path)
    cm, ass = compartmental_model(model, advan, trans)

    assert ass.symbol == S('F')
    assert ass.expression == S('A_CENTRAL') / S('S1')
    assert cm.compartmental_matrix == sympy.Matrix(compmat)
    assert cm.amounts == sympy.Matrix(amounts)
    odes, ics = cm.to_explicit_odes()
    odes = [str(ode) for ode in odes]
    assert odes == strodes
    assert ics == corrics


def test_advan5(testdata):
    model = Model(testdata / 'nonmem' / 'DDMODEL00000130')
    cm, ass = compartmental_model(model, 'ADVAN5', 'TRANS1')
    assert ass.symbol == S('F')
    assert ass.expression == S('A_CMS1')
    assert cm.amounts == sympy.Matrix(
        [S('A_CMS1'), S('A_CMS2'), S('A_COL1'), S('A_INTM'), S('A_INTM2'), S('A_OUTPUT')]
    )
    compmat = sympy.Matrix(
        [
            [-S('K12') - S('K10') - S('K14'), S('K21'), S('K31'), 0, 0, 0],
            [S('K12'), -S('K21') - S('K25'), 0, 0, 0, 0],
            [0, 0, -S('K31') - S('K30'), S('K43'), 0, 0],
            [S('K14'), 0, 0, -S('K43') - S('K40') - S('K45'), S('K54'), 0],
            [0, S('K25'), 0, S('K45'), -S('K54'), 0],
            [S('K10'), 0, S('K30'), S('K40'), 0, 0],
        ]
    )
    assert cm.compartmental_matrix == compmat
