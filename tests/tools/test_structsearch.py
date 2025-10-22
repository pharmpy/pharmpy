from functools import partial

import pytest

from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.modeling import (
    add_metabolite,
    add_peripheral_compartment,
    create_rng,
    set_description,
    set_name,
)
from pharmpy.tools.external.results import parse_modelfit_results
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
from pharmpy.tools.structsearch.tool import (
    categorize_drug_metabolite_model_entries,
    create_result_tables,
    create_workflow,
    validate_input,
)
from pharmpy.workflows import ModelEntry, ModelfitResults, Workflow

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
    model = _add_random_dvids(model)
    models = create_qss_models(model, ests, {'target': 3, 'complex': 2})
    assert len(models) == 8
    assert models[0].dependent_variables == {
        Expr.symbol('Y'): 1,
        Expr.symbol('Y_TARGET'): 3,
        Expr.symbol('Y_COMPLEX'): 2,
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
    model = _add_random_dvids(model)

    models = create_remaining_models(model, ests, 2, None)
    assert len(models) == 12
    models = create_remaining_models(model, ests, 2, dv_types={'drug': 1, 'complex': 3})
    assert len(models) == 11
    for m in models:
        assert 2 not in m.dataset['DVID'].unique()
    models = create_remaining_models(model, ests, 2, dv_types={'drug': 1, 'target': 2})
    assert len(models) == 7
    for m in models:
        assert 3 not in m.dataset['DVID'].unique()
    models = create_remaining_models(
        model, ests, 2, dv_types={'drug': 1, 'target': 2, 'target_tot': 3}
    )
    assert len(models) == 7
    with pytest.raises(ValueError, match='`dv_types` must contain more than 1 dv type'):
        create_remaining_models(model, ests, 2, dv_types={'drug': 1})


def test_create_remaining_models_multiple_dvs(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = _add_random_dvids(model)
    models = create_remaining_models(model, ests, 2, {'target': 2, 'complex': 3})
    assert len(models) == 12


def test_pkpd(load_model_for_test, testdata):
    search_space = (
        "DIRECTEFFECT([LINEAR,EMAX,SIGMOID]);"
        "EFFECTCOMP([LINEAR,EMAX,SIGMOID]);"
        "INDIRECTEFFECT([LINEAR,EMAX,SIGMOID],*)"
    )
    model = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    res = parse_modelfit_results(model, testdata / "nonmem" / "pheno.mod")
    ests = res.parameter_estimates
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
    res = parse_modelfit_results(model, testdata / "nonmem" / "pheno.mod")
    search_space = "METABOLITE([PSC, BASIC]);PERIPHERALS([0,1], MET)"
    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, search_space
    )
    assert base_model_description == "METABOLITE_BASIC;PERIPHERALS(0, METABOLITE)"
    assert len(candidate_tasks) == 4

    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, "METABOLITE([PSC, BASIC])"
    )
    assert base_model_description == "METABOLITE_BASIC"
    assert len(candidate_tasks) == 2

    wb, candidate_tasks, base_model_description = create_drug_metabolite_models(
        model, res, "METABOLITE(BASIC);PERIPHERALS([0,1], MET)"
    )
    assert base_model_description == "METABOLITE_BASIC;PERIPHERALS(0, METABOLITE)"
    assert len(candidate_tasks) == 2


def test_categorize_drug_metabolite_model_entries(
    load_model_for_test, testdata, model_entry_factory
):
    model_start = load_model_for_test(testdata / "nonmem" / "pheno_pd.mod")
    res_start = parse_modelfit_results(model_start, testdata / "nonmem" / "pheno.mod")
    me_start = ModelEntry(model_start, modelfit_results=res_start)

    base_description = 'base'
    funcs = [partial(add_metabolite), partial(add_peripheral_compartment, name='METABOLITE')]

    model = model_start
    candidates = []
    for i, func in enumerate(funcs):
        model = func(model=model)
        model = set_name(model, f'run{i}')
        candidates.append(model)

    mes = model_entry_factory(candidates, parent=model_start)

    base_model_entry, res_model_entries = categorize_drug_metabolite_model_entries(
        me_start, mes, base_description
    )

    assert base_model_entry.model == model_start
    assert len(res_model_entries) == len(funcs)

    base = candidates.pop(0)
    base = set_description(base, base_description)
    candidates.append(base)

    mes = model_entry_factory(candidates, parent=model_start)

    base_model_entry, res_model_entries = categorize_drug_metabolite_model_entries(
        me_start, mes, base_description
    )

    assert base_model_entry.model != model_start
    assert base_model_entry.model.name.startswith('run')
    assert len(res_model_entries) == len(funcs) - 1

    with pytest.raises(ValueError):
        categorize_drug_metabolite_model_entries(None, mes, base_description)


def test_create_workflow_pkpd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    results = parse_modelfit_results(model, testdata / 'nonmem' / 'pheno_pd.mod')
    assert isinstance(create_workflow(model=model, results=results, type='pkpd'), Workflow)


def test_create_workflow_tmdd(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    results = parse_modelfit_results(model, testdata / 'nonmem' / 'pheno_pd.mod')
    assert isinstance(create_workflow(model=model, results=results, type='tmdd'), Workflow)


def test_create_workflow_drug_metabolite(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_pd.mod')
    results = parse_modelfit_results(model, testdata / 'nonmem' / 'pheno_pd.mod')
    assert isinstance(
        create_workflow(model=model, results=results, type='drug_metabolite'), Workflow
    )


def test_create_result_tables(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = parse_modelfit_results(model_start, testdata / 'nonmem' / 'pheno.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    qss_models = create_qss_models(model_start, res_start.parameter_estimates, None)
    qss_entries = model_entry_factory(qss_models)
    qss_best = min(qss_entries, key=lambda me: me.modelfit_results.ofv)

    remaining_models = create_remaining_models(
        model_start, qss_best.modelfit_results.parameter_estimates, 0, None, len(qss_entries)
    )
    remaining_entries = model_entry_factory(remaining_models)

    tables = create_result_tables([[me_start], qss_entries, remaining_entries])

    model_entries = [me_start] + qss_entries + remaining_entries
    summary_models = tables['summary_models']
    assert len(summary_models) == len(model_entries)
    steps = list(summary_models.index.get_level_values('step'))
    assert set(steps) == {0, 1, 2}


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
            dict(type='pkpd', search_space='DIRECTEFFECT(LINEAR)', dv_types={'drug': 1}),
            ValueError,
            'Invalid argument "dv_types" for PKPD models.',
        ),
        (
            dict(type='drug_metabolite', search_space='METABOLITE(BASIC)', dv_types={'drug': 1}),
            ValueError,
            'Invalid argument "dv_types" for drug metabolite models.',
        ),
        (
            dict(type='drug_metabolite', search_space='METABOLITE(BASIC)', met_init=1),
            ValueError,
            'Invalid arguments "b_init", "emax_init", "ec50_init" and "met_init" for drug metabolite models.',
        ),
        (
            dict(type="pkpd"),
            ValueError,
            'Argument search_space need to be specified.',
        ),
        (
            dict(type='pkpd', search_space='DIRECTEFFECT(LINEAR)', extra_model="TEMP"),
            ValueError,
            'Invalid argument "extra_model" for PKPD models.',
        ),
        (
            dict(type='pkpd', search_space='DIRECTEFFECT(LINEAR)', extra_model_results="TEMP"),
            ValueError,
            'Invalid argument "extra_model_results" for PKPD models.',
        ),
        (dict(type="drug_metabolite"), ValueError, 'Argument search_space need to be specified.'),
        (
            dict(type="drug_metabolite", search_space="METABOLITE(BASIC)", extra_model="TEMP"),
            ValueError,
            'Invalid argument "extra_model" for drug metabolite models.',
        ),
        (
            dict(
                type="drug_metabolite", search_space="METABOLITE(BASIC)", extra_model_results="TEMP"
            ),
            ValueError,
            'Invalid argument "extra_model_results" for drug metabolite models.',
        ),
        (dict(type="tmdd", dv_types={'drug_tot': 1, 'target_tot': 2, 'complex': 3}), None, ""),
        (dict(type='tmdd', dv_types={'drug': 1, 'target_tot': 2, 'complex': 3}), None, ""),
        (dict(type='tmdd', dv_types={'drug': 1, 'target': 2, 'complex': 3}), None, ""),
        (
            dict(type='tmdd', dv_types={'drug': 1, 'target': 1, 'complex': 2}),
            ValueError,
            'Values must be unique.',
        ),
        (
            dict(type='tmdd', dv_types={'target': 1, 'complex': 2}),
            ValueError,
            'Only drug can have DVID = 1. Please choose another DVID.',
        ),
    ],
)
def test_validation(tmp_path, load_model_for_test, testdata, arguments, exception, match):
    kwargs = {**arguments}
    kwargs['results'] = ModelfitResults()
    model = load_model_for_test(testdata / "nonmem" / "pheno.mod")
    kwargs['model'] = model
    if "extra_model" in kwargs.keys():
        kwargs["extra_model"] = model
    if "extra_model_results" in kwargs.keys():
        res = parse_modelfit_results(model, testdata / "nonmem" / "pheno.mod")
        kwargs["extra_model_results"] = res

    if exception is not None:
        with pytest.raises(exception, match=match):
            validate_input(**kwargs)
    else:
        validate_input(**kwargs)


def _add_random_dvids(model):
    df = model.dataset.copy()
    rng = create_rng(23)
    df['DVID'] = np.array([rng.integers(1, 6) for i in range(len(df))])
    model = model.replace(dataset=df)
    di = model.datainfo
    col = di['DVID'].replace(type='dvid')
    di = di.set_column(col)
    return model.replace(datainfo=di)
