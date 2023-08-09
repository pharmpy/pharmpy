import pytest

from pharmpy.deps import sympy
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    output,
)
from pharmpy.modeling import add_effect_compartment, set_direct_effect


def S(x):
    return sympy.Symbol(x)


@pytest.mark.parametrize(
    'pd_model',
    [('baseline'), ('linear'), ('Emax'), ('sigmoid'), ('step'), ('loglin')],
)
def test_set_direct_effect(load_model_for_test, pd_model, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    conc = model.statements.ode_system.central_compartment.amount / S("V")
    _test_effect_models(set_direct_effect(model, pd_model), pd_model, conc)


@pytest.mark.parametrize(
    'pd_model',
    [('baseline'), ('linear'), ('Emax'), ('sigmoid'), ('step'), ('loglin')],
)
def test_add_effect_compartment(load_model_for_test, pd_model, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    conc_e = S("A_EFFECT")
    ke0 = S("KE0")
    central_amount = sympy.Function("A_CENTRAL")(S('t'))
    comp_e = Compartment.create("EFFECT", input=ke0 * central_amount / S("V"))

    model1 = add_effect_compartment(model, "baseline")
    compartments = CompartmentalSystemBuilder(model1.statements.ode_system)
    odes = model1.statements.ode_system
    assert odes.find_compartment("EFFECT") == comp_e
    assert odes.zero_order_inputs[0] == 0
    assert odes.zero_order_inputs[1] == ke0 * central_amount / S("V")
    assert odes.get_compartment_outflows("EFFECT")[0][0] == output
    assert odes.get_compartment_outflows("EFFECT")[0][1] == ke0
    assert CompartmentalSystem(compartments).compartment_names == ['CENTRAL', 'EFFECT']

    _test_effect_models(add_effect_compartment(model, pd_model), pd_model, conc_e)


def _test_effect_models(model, expr, conc):
    e = S("E")
    e0 = S("E0")
    emax = S("E_max")
    ec50 = S("EC_50")

    if expr == 'baseline':
        assert model.statements[0] == Assignment(e0, S("POP_E0"))
        assert model.statements.after_odes[-2] == Assignment(e, e0)
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
    elif expr == 'linear':
        assert model.statements[1] == Assignment(e0, S("POP_E0"))
        assert model.statements[0] == Assignment(S("Slope"), S("POP_Slope"))
        assert model.statements.after_odes[-2] == Assignment(e, e0 + S("Slope") * conc)
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
    elif expr == "Emax":
        assert model.statements[0] == Assignment(ec50, S("POP_EC_50"))
        assert model.statements[2] == Assignment(e0, S("POP_E0"))
        assert model.statements[1] == Assignment(emax, S("POP_E_max"))
        assert model.statements.after_odes[-2] == Assignment(e, e0 + (emax * conc) / (ec50 + conc))
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
    elif expr == "sigmoid":
        assert model.statements[0] == Assignment(S("n"), S("POP_n"))
        assert model.statements[1] == Assignment(ec50, S("POP_EC_50"))
        assert model.statements[3] == Assignment(e0, S("POP_E0"))
        assert model.statements[2] == Assignment(emax, S("POP_E_max"))
        assert model.statements.after_odes[-2] == Assignment(
            e,
            sympy.Piecewise(
                (e0 + ((emax * conc ** S("n")) / (ec50 ** S("n") + conc ** S("n"))), conc > 0),
                (e0, True),
            ),
        )
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
        assert model.parameters["POP_n"].init == 1
    elif expr == "step":
        assert model.statements[1] == Assignment(e0, S("POP_E0"))
        assert model.statements[0] == Assignment(emax, S("POP_E_max"))
        assert model.statements.after_odes[-2] == Assignment(
            e, sympy.Piecewise((e0, conc <= 0), (e0 + emax, True))
        )
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
    elif expr == "loglin":
        assert model.statements[1] == Assignment(e0, S("POP_E0"))
        assert model.statements[0] == Assignment(S("slope"), S("POP_slope"))
        assert model.statements.after_odes[-2] == Assignment(
            e, S("slope") * sympy.log(conc + sympy.exp(e0 / S("slope")))
        )
        assert model.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
