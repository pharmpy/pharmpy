import pandas as pd
import pytest
import sympy

from pharmpy.tools import read_modelfit_results
from pharmpy.tools.structsearch.drugmetabolite import create_drug_metabolite_models
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
from pharmpy.tools.structsearch.tool import create_workflow, validate_input
from pharmpy.workflows import Workflow

ests = pd.Series(
    {
        'POP_R_0': 1.0,
        'IIV_R_0': 6.0,
        'POP_KDC': 2.0,
        'POP_KINT': 3.0,
        'POP_KDEG': 4.0,
        'POP_VC': 5.0,
    }
)


def test_create_qss_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_qss_models(model, ests, None)
    assert len(models) == 8


def test_create_qss_models_multiple_dvs(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_qss_models(model, ests, {'target': 3, 'complex': 2})
    assert len(models) == 8
    assert models[0].dependent_variables == {
        sympy.Symbol('Y'): 1,
        sympy.Symbol('Y_TARGET'): 3,
        sympy.Symbol('Y_COMPLEX'): 2,
    }


def test_create_wagner_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_wagner_model(model, ests, None)
    assert len(models) == 1


def test_create_mmapp_model(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_mmapp_model(model, ests, None)
    assert len(models) == 1


def test_create_crib_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_crib_models(model, ests, None)
    assert len(models) == 2


def test_create_full_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_full_models(model, ests, None)
    assert len(models) == 4


def test_create_ib_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_ib_models(model, ests, None)
    assert len(models) == 2


def test_create_cr_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_cr_models(model, ests, None)
    assert len(models) == 2


def test_create_remaining_models(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_remaining_models(model, ests, 2, None)
    assert len(models) == 12


def test_create_remaining_models_multiple_dvs(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    models = create_remaining_models(model, ests, 2, {'target': 2, 'complex': 3})
    assert len(models) == 12


def test_pkpd(load_model_for_test, testdata):
    search_space = "DIRECTEFFECT(*); EFFECTCOMP(*); INDIRECTEFFECT(*,*)"
    res = read_modelfit_results(testdata / "nonmem" / "pheno.mod")
    ests = res.parameter_estimates
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    pkpd_models = create_pkpd_models(
        model, search_space, b_init=5.75, ests=ests, emax_init=2.0, ec50_init=1.0, met_init=0.5
    )

    assert len(pkpd_models) == 12
    assert pkpd_models[0].name == "structsearch_run1"
    assert pkpd_models[1].name == "structsearch_run2"
    assert pkpd_models[1].parameters['POP_B'].init == 5.75
    assert pkpd_models[1].parameters['POP_E_MAX'].init == 2.0
    assert pkpd_models[1].parameters['POP_E_MAX'].fix is False
    assert pkpd_models[1].parameters['POP_EC_50'].init == 1.0
    assert pkpd_models[1].parameters['POP_EC_50'].fix is False
    assert pkpd_models[3].parameters['POP_MET'].init == 0.5
    assert pkpd_models[6].parameters['POP_MET'].init == 0.5

    models3 = create_pkpd_models(model, search_space)
    assert models3[1].parameters['POP_E_MAX'].init == 0.1
    assert models3[1].parameters['POP_E_MAX'].fix is False
    assert models3[1].parameters['POP_EC_50'].init == 0.1
    assert models3[1].parameters['POP_EC_50'].fix is False


def test_drug_metabolite(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    res = read_modelfit_results(testdata / "nonmem" / "pheno.mod")
    search_space = "METABOLITE([PSC, BASIC]);PERIPHERALS([0,1], MET)"
    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, search_space
    )
    assert base_model_description == "METABOLITE_BASIC;PERIPHERALS(0)"
    assert len(candidate_tasks) == 4

    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, "METABOLITE([PSC, BASIC])"
    )
    assert base_model_description == "METABOLITE_BASIC"
    assert len(candidate_tasks) == 2

    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, "METABOLITE(BASIC);PERIPHERALS([0,1], MET)"
    )
    assert base_model_description == "METABOLITE_BASIC;PERIPHERALS(0)"
    assert len(candidate_tasks) == 2


def test_create_workflow():
    assert isinstance(create_workflow('pkpd'), Workflow)


def test_create_workflow_pkpd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('pkpd', model=model), Workflow)


def test_create_workflow_tmdd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('pkpd', model=model), Workflow)


def test_create_workflow_drug_metabolite(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    assert isinstance(create_workflow('drug_metabolite', model=model), Workflow)


@pytest.mark.parametrize(
    ('arguments', 'exception', 'match'),
    [
        (
            dict(type='tmdd', emax_init=0.2),
            ValueError,
            'Invalid arguments "b_init", "emax_init", "ec50_init" and "met_init" for TMDD models.',
        ),
        (
            dict(type='a'),
            ValueError,
            'Invalid `type`',
        ),
        (
            dict(type='tmdd', search_space='ABSORPTION'),
            ValueError,
            'Invalid argument "search_space" for TMDD models.',
        ),
        (
            dict(type='pkpd', dv_types={'drug': 1}),
            ValueError,
            'Invalid argument "dv_types" for PKPD models.',
        ),
        (
            dict(type='drug_metabolite', dv_types={'drug': 1}),
            ValueError,
            'Invalid argument "dv_types" for drug metabolite models.',
        ),
        (
            dict(type='drug_metabolite', met_init=1),
            ValueError,
            'Invalid arguments "b_init", "emax_init", "ec50_init" and "met_init" for drug metabolite models.',
        ),
    ],
)
def test_validation(tmp_path, load_model_for_test, testdata, arguments, exception, match):
    model = load_model_for_test(testdata / "nonmem" / "pheno.mod")
    res = read_modelfit_results(testdata / "nonmem" / "pheno.mod")

    kwargs = {**arguments}
    with pytest.raises(exception, match=match):
        validate_input(**kwargs)

    with pytest.raises(exception, match='Invalid argument "extra_model" for PKPD models.'):
        validate_input(type='pkpd', extra_model=model)
    with pytest.raises(exception, match='Invalid argument "extra_model_results" for PKPD models.'):
        validate_input(type='pkpd', extra_model_results=res)
    with pytest.raises(
        exception, match='Invalid argument "extra_model" for drug metabolite models.'
    ):
        validate_input(type='drug_metabolite', extra_model=model)
    with pytest.raises(
        exception, match='Invalid argument "extra_model_results" for drug metabolite models.'
    ):
        validate_input(type='drug_metabolite', extra_model_results=res)

    validate_input(type='tmdd', dv_types={'drug_tot': 1, 'target_tot': 2, 'complex': 3})
    validate_input(type='tmdd', dv_types={'drug': 1, 'target_tot': 2, 'complex': 3})
    validate_input(type='tmdd', dv_types={'drug': 1, 'target': 2, 'complex': 3})

    with pytest.raises(exception, match='Values must be unique.'):
        validate_input(type='tmdd', dv_types={'drug': 1, 'target': 1, 'complex': 2})
    with pytest.raises(exception, match='Only drug can have DVID = 1. Please choose another DVID.'):
        validate_input(type='tmdd', dv_types={'target': 1, 'complex': 2})
