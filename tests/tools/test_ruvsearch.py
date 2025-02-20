import shutil
from dataclasses import replace

import pytest

from pharmpy.basic import Expr
from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import (
    has_combined_error_model,
    has_proportional_error_model,
    remove_parameter_uncertainty_step,
    set_combined_error_model,
    set_iiv_on_ruv,
    transform_blq,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.ruvsearch.results import calculate_results, psn_resmod_results
from pharmpy.tools.ruvsearch.tool import (
    _change_proportional_model,
    _create_base_model,
    _create_best_model,
    _create_combined_model,
    _create_dataset,
    _create_iiv_on_ruv_model,
    _create_power_model,
    _create_time_varying_model,
    create_result_tables,
    create_workflow,
    validate_input,
)
from pharmpy.workflows import ModelEntry, Workflow


def test_filter_dataset(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem/pheno_pd.mod')
    res = read_modelfit_results(testdata / 'nonmem/pheno_pd.mod')
    indices = model.dataset.index[model.dataset['DVID'] == 2].tolist()
    model_entry = ModelEntry.create(model, modelfit_results=res)
    df = _create_dataset(model_entry, dv=2)
    expected_cwres = [-1.15490, 0.95703, -0.85365, 0.42327]
    assert df['DV'].tolist() == expected_cwres
    assert df['IPRED'].tolist() == res.predictions['CIPREDI'].iloc[indices].tolist()
    assert df['ID'].tolist() == model.dataset['ID'].iloc[indices].tolist()

    ci_dvid = model.datainfo.typeix['dvid'][0].replace(type='unknown')
    di = model.datainfo.set_column(ci_dvid)
    model = model.replace(datainfo=di)
    model_entry = ModelEntry.create(model, modelfit_results=res)
    df = _create_dataset(model_entry, dv=2)
    expected_cwres = [-1.15490, 0.95703, -0.85365, 0.42327]
    assert df['DV'].tolist() == expected_cwres
    assert df['IPRED'].tolist() == res.predictions['CIPREDI'].iloc[indices].tolist()
    assert df['ID'].tolist() == model.dataset['ID'].iloc[indices].tolist()


def test_resmod_results(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir1')
    assert list(res.cwres_models['dOFV']) == [
        -1.31,
        -3.34,
        -13.91,
        -18.54,
        -8.03,
        -4.20,
        -0.25,
        -1.17,
        -0.00,
        -0.09,
        -2.53,
        -3.12,
        -3.60,
        -25.62,
        -7.66,
        -0.03,
        -5.53,
    ]


def test_resmod_results_dvid(testdata):
    res = psn_resmod_results(testdata / 'psn' / 'resmod_dir2')
    df = res.cwres_models
    assert df['dOFV'].loc[1, '1', 'autocorrelation'] == -0.74
    assert df['dOFV'].loc[1, 'sum', 'tdist'] == -35.98


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    remove_parameter_uncertainty_step(model)
    assert isinstance(create_workflow(model=model, results=results), Workflow)


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = remove_parameter_uncertainty_step(model)
    validate_input(model=model, results=res)


def test_create_dataset(load_model_for_test, testdata, tmp_path):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model_entry = ModelEntry.create(model, modelfit_results=res)
    df = _create_dataset(model_entry, dv=None)

    assert len(df) == 1006
    assert (df['DV'] != 0).all()

    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        # Introduce 0 in CWRES to mimic rows BLQ
        with open('mytab') as f:
            mytab_new = f.read().replace('-2.4366E+00', '0.0000E+00')

        with open('mytab', 'w') as f:
            f.write(mytab_new)

        model = load_model_for_test('mox3.mod')
        res = read_modelfit_results('mox3.mod')

        model = transform_blq(model, method='m3', lloq=0.05)
        model_entry = ModelEntry.create(model, modelfit_results=res)

        df = _create_dataset(model_entry, dv=None)

        assert len(df) == 1005
        assert (df['DV'] != 0).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_create_result_tables(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    model_1 = set_combined_error_model(model_start)
    model_1 = model_1.replace(name='model_1')
    model_2 = set_iiv_on_ruv(model_1)
    model_2 = model_2.replace(name='model_2')

    candidate_entries = model_entry_factory([model_1, model_2])
    model_entries = [me_start] + candidate_entries
    tables = create_result_tables(model_entries, cutoff=3.84, strictness='minimization_successful')

    summary_models = tables['summary_models']
    assert len(summary_models) == len(model_entries)
    steps = list(summary_models.index.get_level_values('step'))
    assert steps == [0, 1, 2]

    summary_tool = tables['summary_tool']
    assert len(summary_tool) == len(model_entries)
    steps = list(summary_tool.index.get_level_values('step'))
    assert steps == [0, 1, 2]
    n_params = summary_tool['n_params'].values
    assert list(n_params) == [5, 6, 7]


def test_change_proportional_model(load_model_for_test, testdata):
    model_start = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model_start = set_combined_error_model(model_start)
    assert has_combined_error_model(model_start)
    me_start = ModelEntry.create(model_start, modelfit_results=None)

    me_base = _change_proportional_model(me_start)
    assert me_base.model.description.startswith('Input model')
    assert has_proportional_error_model(me_base.model)


@pytest.mark.parametrize(
    ('func', 'kwargs', 'description', 'y_str'),
    [
        (
            _create_iiv_on_ruv_model,
            dict(),
            'IIV_on_RUV_1',
            'theta + eta_base + epsilon*exp(ETA_RV1)',
        ),
        (
            _create_power_model,
            dict(),
            'power_1',
            'theta + eta_base + IPRED**power1*epsilon',
        ),
        (
            _create_time_varying_model,
            {'groups': 4, 'i': 1},
            'time_varying1_1',
            'epsilon*time_varying + eta_base + theta',
        ),
        (
            _create_combined_model,
            dict(),
            'combined_1',
            'theta + eta_base + epsilon_p + epsilon_a/IPRED',
        ),
    ],
)
def test_create_models(load_model_for_test, testdata, func, kwargs, description, y_str):
    model_start = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    me_base = _create_base_model(me_start, current_iteration=1, dv=None)
    assert me_base.model.description == 'base_1'
    assert me_base.model.statements.find_assignment('Y').expression == Expr(
        'theta + eta_base + epsilon'
    )

    me_cand = func(me_base, current_iteration=1, **kwargs)
    assert me_cand.model.description == description

    y_expr = me_cand.model.statements.find_assignment('Y').expression
    if y_expr.is_piecewise():
        y_expr = y_expr.args[0][0]

    assert y_expr == Expr(y_str)


@pytest.mark.parametrize(
    ('func', 'kwargs', 'best_model_name_ref', 'y_str'),
    [
        (
            _create_iiv_on_ruv_model,
            dict(),
            'IIV_on_RUV',
            'EPS_1*A_CENTRAL(t)*exp(ETA_RV1)/VC + A_CENTRAL(t)/VC',
        ),
        (
            _create_power_model,
            dict(),
            'power',
            'EPS_1*(A_CENTRAL(t)/VC)**power1 + A_CENTRAL(t)/VC',
        ),
        (
            _create_time_varying_model,
            {'groups': 4, 'i': 1},
            'time_varying1',
            'EPS_1*time_varying*A_CENTRAL(t)/VC + A_CENTRAL(t)/VC',
        ),
        (
            _create_combined_model,
            dict(),
            'combined',
            'epsilon_a + epsilon_p*A_CENTRAL(t)/VC + A_CENTRAL(t)/VC',
        ),
    ],
)
def test_create_best_model(
    load_model_for_test, testdata, model_entry_factory, func, kwargs, best_model_name_ref, y_str
):
    model_start = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    model_base = _create_base_model(me_start, current_iteration=1, dv=None).model
    me_base = model_entry_factory([model_base])[0]
    model_cwres = func(me_base, current_iteration=1, **kwargs).model
    me_cwres = model_entry_factory([model_cwres], ref_val=me_base.modelfit_results.ofv - 3.84)[0]
    res = calculate_results([me_base, me_cwres])
    me_best, best_model_name = _create_best_model(
        me_start, res, current_iteration=1, dv=None, groups=6
    )

    assert me_best.model.name == 'best_ruvsearch_1'

    y_expr = me_best.model.statements.find_assignment('Y').expression
    if y_expr.is_piecewise():
        y_expr = y_expr.args[0][0]
    assert y_expr == Expr(y_str)
    assert best_model_name == best_model_name_ref


def test_create_best_model_no_best(load_model_for_test, testdata, model_entry_factory):
    model_start = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res_start = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    me_start = ModelEntry.create(model_start, modelfit_results=res_start)

    model_base = _create_base_model(me_start, current_iteration=1, dv=None).model
    me_base = model_entry_factory([model_base])[0]
    funcs = [_create_iiv_on_ruv_model, _create_power_model, _create_combined_model]

    cwres_models = [func(me_base, current_iteration=1).model for func in funcs]
    cwres_model_entries = model_entry_factory(cwres_models)
    res = calculate_results([me_base] + cwres_model_entries)

    me_best, best_model_name = _create_best_model(
        me_start, res, current_iteration=1, dv=None, groups=4, cutoff=99999999999
    )
    assert me_best is None and best_model_name is None


@pytest.mark.parametrize(
    ('model_path', 'arguments', 'exception', 'match'),
    [
        (
            None,
            dict(groups=3.1415),
            TypeError,
            'Invalid `groups`',
        ),
        (
            None,
            dict(groups=0),
            ValueError,
            'Invalid `groups`',
        ),
        (
            None,
            dict(p_value='x'),
            TypeError,
            'Invalid `p_value`',
        ),
        (
            None,
            dict(p_value=1.01),
            ValueError,
            'Invalid `p_value`',
        ),
        (
            None,
            dict(skip='ABC'),
            TypeError,
            'Invalid `skip`',
        ),
        (
            None,
            dict(skip=1),
            TypeError,
            'Invalid `skip`',
        ),
        (
            None,
            dict(skip=['IIV_on_RUV', 'power', 'time', 0]),
            TypeError,
            'Invalid `skip`',
        ),
        (
            None,
            dict(skip=['IIV_on_RUV', 'power', 'time']),
            TypeError,
            'Invalid `skip`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
        (
            None,
            dict(max_iter=4),
            ValueError,
            'Invalid `max_iter`',
        ),
        (
            ('nonmem/ruvsearch/mox3.mod',),
            dict(strictness='rse'),
            ValueError,
            '`parameter_uncertainty_method` not set',
        ),
        (
            ('nonmem/pheno_real.mod',),
            dict(),
            ValueError,
            'Invalid `model`: TAD must be a column',
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
        model_path = ('nonmem/ruvsearch/mox3.mod',)
    path = testdata.joinpath(*model_path)
    model = load_model_for_test(path)
    res = read_modelfit_results(path)
    kwargs = {'model': model, 'results': res, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)


def test_validate_input_raises_cwres(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = remove_parameter_uncertainty_step(model)
    modelfit_results = replace(res, residuals=res.residuals.drop(columns=['CWRES']))

    with pytest.raises(ValueError, match="CWRES"):
        validate_input(model=model, results=modelfit_results)


def test_validate_input_raises_cipredi(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = remove_parameter_uncertainty_step(model)
    modelfit_results = replace(res, predictions=res.predictions.drop(columns=['CIPREDI']))

    with pytest.raises(ValueError, match="IPRED"):
        validate_input(model=model, results=modelfit_results)


def test_validate_input_raises_ipred(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    model = remove_parameter_uncertainty_step(model)
    modelfit_results = replace(res, predictions=res.predictions.drop(columns=['IPRED']))

    with pytest.raises(ValueError, match="IPRED"):
        validate_input(model=model, results=modelfit_results)


def test_validate_input_raises_blq(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = transform_blq(model, method='m4', lloq=1.0)

    with pytest.raises(ValueError, match="BLQ"):
        validate_input(model=model, results=res, max_iter=2)


def test_validate_input_raises_dv(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')

    with pytest.raises(ValueError, match="No DVID column"):
        validate_input(model=model, results=res, dv=1)

    di = model.datainfo
    dvid_col = di['IACE'].replace(type='dvid')
    di = di.set_column(dvid_col)
    model = model.replace(datainfo=di)
    with pytest.raises(ValueError, match="No IACE = 10"):
        validate_input(model=model, results=res, dv=10)

    df = model.dataset.rename(columns={'IACE': 'DVID'})
    model = model.replace(dataset=df)
    with pytest.raises(ValueError, match="No DVID = 10"):
        validate_input(model=model, results=res, dv=10)
