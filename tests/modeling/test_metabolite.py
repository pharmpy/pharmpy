from pharmpy.modeling import add_metabolite


def test_add_metabolite(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
