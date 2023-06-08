from pharmpy.modeling import create_joint_distribution, remove_iiv
from pharmpy.tools import read_modelfit_results


def test_nested_update_source(load_model_for_test, pheno_path):
    model = load_model_for_test(pheno_path)
    res = read_modelfit_results(pheno_path)

    model = create_joint_distribution(model, individual_estimates=res.individual_estimates)
    model = model.update_source()

    assert 'IIV_CL_IIV_V' in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'CL')

    model = model.update_source()

    assert '0.031128' in model.model_code
    assert '0.0309626' not in model.model_code

    model = load_model_for_test(pheno_path)

    model = remove_iiv(model, 'V')

    model = model.update_source()

    assert '0.0309626' in model.model_code
    assert '0.031128' not in model.model_code
