from pharmpy.modeling import add_metabolite


def test_add_metabolite(testdata, load_model_for_test):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno_conc.mod')
    model = add_metabolite(model)
    odes = model.statements.ode_system
    assert odes.compartment_names == ['CENTRAL', 'METABOLITE']
    a = model.model_code.split('\n')
    assert a[20] == 'IF (DVID.EQ.1) THEN'
    assert a[21] == '    Y = Y'
    assert a[22] == 'ELSE'
    assert a[23] == '    Y = Y_M'
    assert a[24] == 'END IF'
