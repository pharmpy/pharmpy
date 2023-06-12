from pharmpy.model.external.nlmixr import convert_model


def test_model(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert 'ini' in model.model_code
    assert 'model' in model.model_code


def test_pheno_real(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(nmmodel)
    assert '} else {' in model.model_code


def test_pheno_error_model(testdata, load_model_for_test):
    mmodel = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(mmodel)
    assert "add_error <- 0" in model.model_code
    assert "prop_error <- SIGMA_1_1" in model.model_code
    assert "Y ~ add(add_error) + prop(prop_error)" in model.model_code


def test_prop_alias(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = convert_model(nmmodel)
    assert "prop_error <- RUV_PROP" in model.model_code


def test_remove_sigma(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')
    model = convert_model(nmmodel)
    assert "SIGMA_1_1" not in model.model_code


def test_sigma(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = convert_model(nmmodel)
    assert "RUV_PROP <- c(0.0, 0.1, Inf)" in model.model_code


def test_dataset_modifications(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert "EVID" in model.dataset.columns


def test_w_error_model(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')
    model = convert_model(nmmodel)
    assert 'add_error <- sqrt(THETA_3**2)' in model.model_code
    assert 'prop_error <- sqrt(THETA_4**2)' in model.model_code
    assert "Y ~ add(add_error) + prop(prop_error)" in model.model_code


def test_omega_same(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')
    model = convert_model(nmmodel)
    assert "IOV_CL_7" in model.parameters
    assert "IOV_V_7" in model.parameters
