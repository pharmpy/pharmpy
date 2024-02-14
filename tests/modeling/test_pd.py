import pytest

from pharmpy.basic import Expr
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    output,
)
from pharmpy.modeling import (
    add_effect_compartment,
    add_indirect_effect,
    set_baseline_effect,
    set_direct_effect,
    set_michaelis_menten_elimination,
)


def S(x):
    return Expr.symbol(x)


@pytest.mark.parametrize(
    'pd_model',
    [('linear'), ('emax'), ('sigmoid'), ('step'), ('loglin')],
)
def test_set_direct_effect(load_model_for_test, pd_model, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    conc = model.statements.ode_system.central_compartment.amount / S("VC")
    _test_effect_models(set_direct_effect(model, pd_model), pd_model, conc)


@pytest.mark.parametrize(
    'pd_model',
    [('linear'), ('emax'), ('sigmoid'), ('step'), ('loglin')],
)
def test_add_effect_compartment(load_model_for_test, pd_model, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    conc_e = Expr.function("A_EFFECT", 't')
    ke0 = S("KE0")
    central_amount = Expr.function("A_CENTRAL", S('t'))
    comp_e = Compartment.create("EFFECT", input=ke0 * central_amount / S("VC"))

    model1 = add_effect_compartment(model, "linear")
    compartments = CompartmentalSystemBuilder(model1.statements.ode_system)
    odes = model1.statements.ode_system
    assert odes.find_compartment("EFFECT") == comp_e
    assert odes.zero_order_inputs[0] == 0
    assert odes.zero_order_inputs[1] == ke0 * central_amount / S("VC")
    assert odes.get_compartment_outflows("EFFECT")[0][0] == output
    assert odes.get_compartment_outflows("EFFECT")[0][1] == ke0
    assert CompartmentalSystem(compartments).compartment_names == ['CENTRAL', 'EFFECT']

    _test_effect_models(add_effect_compartment(model, pd_model), pd_model, conc_e)


def _test_effect_models(model, expr, conc):
    e = S("E")
    e0 = S("B")
    emax = S("E_MAX")
    ec50 = S("EC_50")

    if expr == 'linear':
        assert model.statements[1] == Assignment.create(e0, S("POP_B"))
        assert model.statements[0] == Assignment.create(S("SLOPE"), S("POP_SLOPE"))
        assert model.statements.after_odes[-2] == Assignment.create(e, e0 * (1 + S("SLOPE") * conc))
        assert model.statements.after_odes[-1] == Assignment.create(
            S("Y_2"), e + e * S("epsilon_p")
        )
    elif expr == "emax":
        assert model.statements[0] == Assignment.create(ec50, S("POP_EC_50"))
        assert model.statements[2] == Assignment.create(e0, S("POP_B"))
        assert model.statements[1] == Assignment.create(emax, S("POP_E_MAX"))
        assert model.statements.after_odes[-2] == Assignment.create(
            e, e0 * (1 + (emax * conc) / (ec50 + conc))
        )
        assert model.statements.after_odes[-1] == Assignment.create(
            S("Y_2"), e + e * S("epsilon_p")
        )
    elif expr == "sigmoid":
        assert model.statements[0] == Assignment.create(S("N"), S("POP_N"))
        assert model.statements[1] == Assignment.create(ec50, S("POP_EC_50"))
        assert model.statements[3] == Assignment.create(e0, S("POP_B"))
        assert model.statements[2] == Assignment.create(emax, S("POP_E_MAX"))
        assert model.statements.after_odes[-2] == Assignment.create(
            e,
            Expr.piecewise(
                (
                    e0 * (1 + ((emax * conc ** S("N")) / (ec50 ** S("N") + conc ** S("N")))),
                    conc > 0,
                ),
                (e0, True),
            ),
        )
        assert model.statements.after_odes[-1] == Assignment.create(
            S("Y_2"), e + e * S("epsilon_p")
        )
        assert model.parameters["POP_N"].init == 1
    elif expr == "step":
        assert model.statements[1] == Assignment.create(e0, S("POP_B"))
        assert model.statements[0] == Assignment.create(emax, S("POP_E_MAX"))
        assert model.statements.after_odes[-2] == Assignment.create(
            e, Expr.piecewise((e0, conc <= 0), (e0 * (1 + emax), True))
        )
        assert model.statements.after_odes[-1] == Assignment.create(
            S("Y_2"), e + e * S("epsilon_p")
        )
    elif expr == "loglin":
        assert model.statements[1] == Assignment.create(e0, S("POP_B"))
        assert model.statements[0] == Assignment.create(S("SLOPE"), S("POP_SLOPE"))
        assert model.statements.after_odes[-2] == Assignment.create(
            e, S("SLOPE") * (conc + (e0 / S("SLOPE")).exp()).log()
        )
        assert model.statements.after_odes[-1] == Assignment.create(
            S("Y_2"), e + e * S("epsilon_p")
        )


@pytest.mark.parametrize(
    'prod, expr',
    [
        (True, 'linear'),
        (True, 'emax'),
        (True, 'sigmoid'),
        (False, 'linear'),
        (False, 'emax'),
        (False, 'sigmoid'),
    ],
)
def test_indirect_effect(load_model_for_test, testdata, prod, expr):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    model = add_indirect_effect(
        model,
        prod=prod,
        expr=expr,
    )


def test_set_baseline_effect(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    baseline = set_baseline_effect(model)

    e, e0 = S('E'), S('B')
    assert baseline.statements[0] == Assignment.create(e0, S("POP_B"))
    assert baseline.statements.after_odes[-2] == Assignment.create(e, e0)
    assert baseline.statements.after_odes[-1] == Assignment.create(S("Y_2"), e + e * S("epsilon_p"))


def test_pd_michaelis_menten(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    pkpd = set_direct_effect(model, 'linear')
    michaelis_menten = set_michaelis_menten_elimination(model)
    pkpd_mm = set_direct_effect(michaelis_menten, 'linear')
    assert pkpd.statements.find_assignment('E') == pkpd_mm.statements.find_assignment('E')

    pkpd = add_effect_compartment(model, 'linear')
    michaelis_menten = set_michaelis_menten_elimination(model)
    pkpd_mm = add_effect_compartment(michaelis_menten, 'linear')
    assert pkpd.statements.find_assignment('E') == pkpd_mm.statements.find_assignment('E')
