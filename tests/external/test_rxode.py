from pharmpy.model.external.rxode import convert_model


def test_model(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert 'rxode2' in model.model_code
    assert 'thetas' in model.model_code
    assert 'omegas' in model.model_code
    assert 'sigmas' in model.model_code


def test_pheno_real(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(nmmodel)
    assert '} else {' in model.model_code


def test_pheno_error_model(testdata, load_model_for_test):
    mmodel = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    model = convert_model(mmodel)
    assert "Y <- F + SIGMA_1_1*W" in model.model_code


def test_sigma(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    model = convert_model(nmmodel)
    assert "RUV_PROP ~ 0.1" in model.model_code


def test_dataset_modifications(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = convert_model(nmmodel)
    assert "EVID" in model.dataset.columns


def test_w_error_model(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')
    model = convert_model(nmmodel)
    assert 'W <- sqrt(IPRED**2*THETA_4**2 + THETA_3**2)' in model.model_code


def test_omega_same(testdata, load_model_for_test):
    nmmodel = load_model_for_test(testdata / 'nonmem' / 'models' / 'fviii6.mod')
    model = convert_model(nmmodel)
    assert "IOV_CL_7" in model.parameters
    assert "IOV_V_7" in model.parameters
