import sympy

from pharmpy.modeling import set_tmdd


def test_set_tmdd(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = set_tmdd(model, type="full")
    assert model.statements.ode_system.eqs[0].rhs == sympy.sympify(
        'A_CENTRAL(t)*(-A_TARGET(t)*KON - CL/V) + A_COMPLEX(t)*KOFF'
    )
