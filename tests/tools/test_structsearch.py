import pandas as pd

from pharmpy.tools import read_modelfit_results
from pharmpy.tools.structsearch.pkpd import create_pkpd_models
from pharmpy.tools.structsearch.tmdd import (
    create_cr_models,
    create_crib_models,
    create_full_models,
    create_ib_models,
    create_mmapp_model,
    create_qss_models,
    create_remaining_models,
    create_wagner_model,
)
from pharmpy.tools.structsearch.tool import create_workflow
from pharmpy.workflows import Workflow

ests = pd.Series(
    {'POP_R0': 1.0, 'IIV_R0': 6.0, 'POP_KDC': 2.0, 'POP_KINT': 3.0, 'POP_KDEG': 4.0, 'POP_VC': 5.0}
)


def test_create_qss_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_qss_models(model)
    assert len(models) == 8


def test_create_wagner_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_wagner_model(model, ests)
    assert len(models) == 1


def test_create_mmapp_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_mmapp_model(model, ests)
    assert len(models) == 1


def test_create_crib_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_crib_models(model, ests)
    assert len(models) == 2


def test_create_full_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_full_models(model, ests)
    assert len(models) == 4


def test_create_ib_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_ib_models(model, ests)
    assert len(models) == 2


def test_create_cr_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_cr_models(model, ests)
    assert len(models) == 2


def test_create_remaining_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_remaining_models(model, ests)
    assert len(models) == 12


def test_pkpd(load_model_for_test, testdata):
    res = read_modelfit_results(testdata / "nonmem" / "pheno.mod")
    ests = res.parameter_estimates
    e0_init = pd.Series({'POP_E0': 5.75, 'IIV_E0': 0.01, 'sigma': 0.33})
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    pkpd_models = create_pkpd_models(model, e0_init, ests)

    assert len(pkpd_models) == 8
    assert pkpd_models[0].name == "structsearch_run1"
    assert pkpd_models[1].name == "structsearch_run2"
    assert pkpd_models[4].name == "structsearch_run5"
    assert pkpd_models[5].name == "structsearch_run6"
    assert pkpd_models[0].parameters[0].name == 'TVCL'
    assert pkpd_models[5].parameters[0].name == 'TVCL'
    assert pkpd_models[0].parameters[1].name == 'TVV'
    assert pkpd_models[5].parameters[1].name == 'TVV'


def test_create_workflow():
    assert isinstance(create_workflow('oral', 'pkpd'), Workflow)


def test_create_workflow_pkpd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('oral', 'pkpd', model=model), Workflow)


def test_create_workflow_tmdd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('oral', 'pkpd', model=model), Workflow)
