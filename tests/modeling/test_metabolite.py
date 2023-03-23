from pharmpy.modeling import add_metabolite


def test_add_metabolite(pheno_path, load_model_for_test):
    model = load_model_for_test(pheno_path)
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
    a = model.model_code.split('\n')
    assert a[25] == 'IF (DVID.EQ.1) THEN'
    assert a[26] == '    Y = EPS(1)*W + F'
    assert a[27] == 'ELSE'
    assert a[28] == '    Y = CONC_M1 + EPS(1)'
    assert a[29] == 'END IF'
