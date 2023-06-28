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
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    pkpd_models = create_pkpd_models(model, ests)
    assert len(pkpd_models) == 12
    assert pkpd_models[0].name == "direct_effect_baseline"
    assert pkpd_models[1].name == "direct_effect_linear"
    assert pkpd_models[6].name == "effect_compartment_baseline"
    assert pkpd_models[7].name == "effect_compartment_linear"
    assert pkpd_models[0].parameters[0].name == 'TVCL'
    assert pkpd_models[6].parameters[0].name == 'TVCL'
    assert pkpd_models[0].parameters[1].name == 'TVV'
    assert pkpd_models[6].parameters[1].name == 'TVV'


def test_create_workflow():
    assert isinstance(create_workflow('oral', 'pkpd'), Workflow)


def test_create_workflow_pkpd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('oral', 'pkpd', model=model), Workflow)


def test_create_workflow_tmdd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('oral', 'pkpd', model=model), Workflow)
