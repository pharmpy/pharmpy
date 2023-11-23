import pytest
import sympy

from pharmpy.modeling import add_peripheral_compartment, set_mixed_mm_fo_elimination, set_tmdd


def test_full(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="full")
    assert model.statements.ode_system.eqs[0].rhs == sympy.sympify(
        'A_CENTRAL(t)*(-A_TARGET(t)*KON/V - CL/V) + A_COMPLEX(t)*KOFF'
    )


def test_qss(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="qss")
    assert model.statements.ode_system.eqs[0].rhs == sympy.sympify(
        '-A_CENTRAL(t)*CL*LAFREE/V - A_TARGET(t)*KINT*LAFREE/(KD + LAFREE) - CL*LAFREE/V'
    )


def test_qss_1c(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    tmdd_model = set_tmdd(model, type="qss")
    assert (
        str(tmdd_model.statements.ode_system.eqs[0].rhs)
        == """-CL*LAFREE/V1 - KINT*LAFREE*A_TARGET(t)/(KD + LAFREE) - LAFREE*QP1/V1 + QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(tmdd_model.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(tmdd_model.statements.ode_system.eqs[2].rhs)
        == """KSYN*V1 + (-KDEG - LAFREE*(-KDEG + KINT)/(KD + LAFREE))*A_TARGET(t)"""
    )

    model2 = add_peripheral_compartment(model)
    tmdd_model = set_tmdd(model2, type="qss")


def test_cr(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="cr")
    assert model.statements.ode_system.eqs[1].rhs == sympy.sympify(
        '(-KINT - KOFF)*A_COMPLEX(t) + (KON*R_0 - KON*A_COMPLEX(t)/V)*A_CENTRAL(t)'
    )


def test_crib(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="crib")
    assert model.statements.ode_system.eqs[1].rhs == sympy.sympify(
        '-KINT*A_COMPLEX(t) + (KON*R_0 - KON*A_COMPLEX(t)/V)*A_CENTRAL(t)'
    )


def test_ib(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="ib")
    assert model.statements.ode_system.eqs[2].rhs == sympy.sympify(
        '-KDEG*A_TARGET(t) - KON*A_CENTRAL(t)*A_TARGET(t)/V + KSYN*V'
    )


def test_mmapp(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, type="mmapp")
    assert (
        str(model.statements.ode_system.eqs[0].rhs)
        == """QP1*A_PERIPHERAL1(t)/VP1 + (-CL/V1 - KINT*A_TARGET(t)/(KM + A_CENTRAL(t)/V1) - QP1/V1)*A_CENTRAL(t)"""
    )


def test_wagner(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, type="wagner")


def test_tmdd_with_mixed_fo_mm_elimination(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = add_peripheral_compartment(model)
    model = set_mixed_mm_fo_elimination(model)
    qss = set_tmdd(model, 'qss')
    assert qss


def test_wagner_1c(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    wagner = set_tmdd(model, type="wagner")
    assert (
        str(wagner.statements.ode_system.eqs[0].rhs)
        == """-CL*LAFREE/V1 + KINT*LAFREE - KINT*A_CENTRAL(t) - LAFREE*QP1/V1 + QP1*A_PERIPHERAL1(t)/VP1"""
    )
    assert (
        str(wagner.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )

    model = add_peripheral_compartment(model)
    wagner = set_tmdd(model, type="wagner")
    assert (
        str(wagner.statements.ode_system.eqs[1].rhs)
        == """LAFREE*QP1/V1 - QP1*A_PERIPHERAL1(t)/VP1"""
    )


@pytest.mark.parametrize(
    ('model_name', 'dv_types', 'expected'),
    (
        (
            'full',
            {'drug': 1, 'target': 2, 'complex': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2, sympy.Symbol('Y_COMPLEX'): 3},
        ),
        (
            'qss',
            {'drug': 1, 'target': 2, 'complex': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2, sympy.Symbol('Y_COMPLEX'): 3},
        ),
        (
            'ib',
            {'drug': 1, 'target': 2, 'complex': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2, sympy.Symbol('Y_COMPLEX'): 3},
        ),
        ('mmapp', {'drug': 1, 'target': 2}, {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2}),
        ('cr', {'drug': 1, 'complex': 2}, {sympy.Symbol('Y'): 1, sympy.Symbol('Y_COMPLEX'): 2}),
        ('wagner', {'drug': 1, 'complex': 2}, {sympy.Symbol('Y'): 1, sympy.Symbol('Y_COMPLEX'): 2}),
        ('crib', {'drug': 1, 'complex': 2}, {sympy.Symbol('Y'): 1, sympy.Symbol('Y_COMPLEX'): 2}),
        ('crib', {'drug': 1, 'complex': 3}, {sympy.Symbol('Y'): 1, sympy.Symbol('Y_COMPLEX'): 3}),
        (
            'full',
            {'target': 2, 'complex': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2, sympy.Symbol('Y_COMPLEX'): 3},
        ),
        (
            'cr',
            {'drug': 1, 'complex': 2, 'target': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_COMPLEX'): 2},
        ),
        (
            'mmapp',
            {'drug': 1, 'target': 2, 'complex': 3},
            {sympy.Symbol('Y'): 1, sympy.Symbol('Y_TARGET'): 2},
        ),
    ),
    ids=repr,
)
def test_full_multiple_dvs(pheno_path, load_model_for_test, model_name, dv_types, expected):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, model_name, dv_types)
    assert model.dependent_variables == expected
    assert len(model.random_variables.epsilons) > 1
