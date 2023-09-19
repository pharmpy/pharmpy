import sympy

from pharmpy.modeling import add_peripheral_compartment, set_tmdd


def test_full(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="full")
    assert model.statements.ode_system.eqs[0].rhs == sympy.sympify(
        'A_CENTRAL(t)*(-A_TARGET(t)*KON - CL/V) + A_COMPLEX(t)*KOFF'
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
        == """-CL*LCFREE/V1 - KINT*LCFREE*A_TARGET(t)/(KD + LCFREE) - LCFREE*Q/V1 + Q*A_PERIPHERAL1(t)/V2"""
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
    model = set_tmdd(model, type="mmapp")


def test_wagner(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_peripheral_compartment(model)
    model = set_tmdd(model, type="wagner")
