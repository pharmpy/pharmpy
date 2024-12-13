import pytest

from pharmpy.tools import read_modelfit_results
from pharmpy.tools.allometry.tool import (
    add_allometry_on_model,
    create_result_tables,
    create_workflow,
    get_best_model,
    validate_input,
)
from pharmpy.workflows import ModelEntry, Workflow
from pharmpy.workflows.contexts import NullContext


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    assert isinstance(
        create_workflow(model=model, results=results, allometric_variable='WGT'), Workflow
    )


def test_add_allometry_on_model(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)
    me_cand = add_allometry_on_model(
        me_start,
        allometric_variable='WGT',
        reference_value=70,
        parameters=None,
        initials=None,
        lower_bounds=None,
        upper_bounds=None,
        fixed=None,
    )
    assert me_cand.parent.name == 'pheno'
    model_cand = me_cand.model
    assert model_cand.parameters['TVCL'].init != model_start.parameters['TVCL'].init
    assert len(model_cand.parameters) > len(model_start.parameters)


def test_get_best_model(load_model_for_test, testdata):
    ctx = NullContext()
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    model_allometry = model_start.replace(description='allometry')

    me_start = ModelEntry(model_start, modelfit_results=res_start)
    me_allometry = ModelEntry(model_allometry, modelfit_results=res_start)
    best_model, _ = get_best_model(ctx, me_start, me_allometry)
    assert best_model.name == 'final'
    assert best_model.description == 'allometry'

    me_allometry = ModelEntry(model_allometry, modelfit_results=None)
    best_model, _ = get_best_model(ctx, me_start, me_allometry)
    assert best_model.name == 'final'
    assert best_model.description != 'allometry'


def test_create_result_tables(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    model_allometry = model_start.replace(name='allometry', description='allometry')

    me_start = ModelEntry(model_start, modelfit_results=res_start)
    me_allometry = model_entry_factory([model_allometry])[0]
    summods, _ = create_result_tables(me_start, me_allometry)
    assert len(summods) == 2


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, results=results, allometric_variable='WGT')


def test_validate_input_with_model_and_parameters(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    validate_input(model=model, results=results, allometric_variable='WGT', parameters=['CL', 'V'])


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (None, dict(allometric_variable=1), TypeError, 'Invalid `allometric_variable`'),
        (None, dict(reference_value=[]), TypeError, 'Invalid `reference_value`'),
        (None, dict(parameters='CL'), TypeError, 'Invalid `parameters`'),
        (None, dict(initials=0.1), TypeError, 'Invalid `initials`'),
        (None, dict(lower_bounds=0.0001), TypeError, 'Invalid `lower_bounds`'),
        (None, dict(upper_bounds=1000), TypeError, 'Invalid `upper_bounds`'),
        (None, dict(fixed=1), TypeError, 'Invalid `fixed`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(allometric_variable='WT'),
            ValueError,
            'Invalid `allometric_variable`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(allometric_variable='WGT', parameters=['K']),
            ValueError,
            'Invalid `parameters`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    exception,
    match,
):
    if not model_path:
        model_path = ('nonmem/pheno.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    results = read_modelfit_results(path)

    kwargs = {'model': model, 'results': results, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)
