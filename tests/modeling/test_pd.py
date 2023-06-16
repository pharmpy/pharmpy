from pharmpy.deps import sympy
from pharmpy.model import Assignment
from pharmpy.modeling import set_direct_effect


def S(x):
    return sympy.Symbol(x)


def test_set_direct_effect(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    conc = model.statements.ode_system.central_compartment.amount / S("V")

    e = S("E")
    e0 = S("E0")
    emax = S("E_max")
    ec50 = S("EC_50")

    model1 = set_direct_effect(model, 'baseline')
    assert model1.statements[0] == Assignment(e0, S("POP_E0"))
    assert model1.statements.after_odes[-2] == Assignment(e, e0)
    assert model1.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))

    model2 = set_direct_effect(model, 'linear')
    assert model2.statements[1] == Assignment(e0, S("POP_E0"))
    assert model2.statements[0] == Assignment(S("S"), S("POP_S"))
    assert model2.statements.after_odes[-2] == Assignment(e, e0 + S("S") * conc)
    assert model2.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))

    model3 = set_direct_effect(model, "Emax")
    assert model3.statements[0] == Assignment(ec50, S("POP_EC_50"))
    assert model3.statements[2] == Assignment(e0, S("POP_E0"))
    assert model3.statements[1] == Assignment(emax, S("POP_E_max"))
    assert model3.statements.after_odes[-2] == Assignment(e, e0 + (emax * conc) / (ec50 + conc))
    assert model3.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))

    model4 = set_direct_effect(model, "sigmoid")
    assert model4.statements[0] == Assignment(S("n"), S("POP_n"))
    assert model4.statements[1] == Assignment(ec50, S("POP_EC_50"))
    assert model4.statements[3] == Assignment(e0, S("POP_E0"))
    assert model4.statements[2] == Assignment(emax, S("POP_E_max"))
    assert model4.statements.after_odes[-2] == Assignment(
        e, (emax * conc ** S("n")) / (ec50 ** S("n") + conc ** S("n"))
    )
    assert model4.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
    assert model4.parameters["POP_n"].init == 1

    model5 = set_direct_effect(model, "step")
    assert model5.statements[1] == Assignment(e0, S("POP_E0"))
    assert model5.statements[0] == Assignment(emax, S("POP_E_max"))
    assert model5.statements.after_odes[-2] == Assignment(
        e, sympy.Piecewise((0, conc < 0), (emax, True))
    )
    assert model5.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))

    model6 = set_direct_effect(model, "loglin")
    assert model6.statements[1] == Assignment(e0, S("POP_E0"))
    assert model6.statements[0] == Assignment(S("m"), S("POP_m"))
    assert model6.statements.after_odes[-2] == Assignment(
        e, S("m") * sympy.log(conc + sympy.exp(e0 / S("m")))
    )
    assert model6.statements.after_odes[-1] == Assignment(S("Y_2"), e + e * S("epsilon_p"))
