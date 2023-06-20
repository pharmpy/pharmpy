import shutil
from dataclasses import replace

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import remove_covariance_step, transform_blq
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.ruvsearch.results import psn_resmod_results
from pharmpy.tools.ruvsearch.tool import _create_dataset, create_workflow, validate_input
from pharmpy.workflows import Workflow


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


def test_create_workflow():
    assert isinstance(create_workflow(), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    remove_covariance_step(model)
    assert isinstance(create_workflow(model=model), Workflow)


def test_validate_input():
    validate_input()


def test_validate_input_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = model.replace(modelfit_results=res)
    model = remove_covariance_step(model)
    validate_input(model=model)


def test_create_dataset(load_model_for_test, testdata, tmp_path):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = model.replace(modelfit_results=res)

    df = _create_dataset(model)

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

        model = model.replace(modelfit_results=res)

        model = transform_blq(model, method='m3', lloq=0.05)

        df = _create_dataset(model)

        assert len(df) == 1005
        assert (df['DV'] != 0).all()


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
            ValueError,
            'Invalid `skip`',
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
    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    kwargs = {'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)


def test_validate_input_raises_modelfit_results(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model = model.replace(modelfit_results=None)

    with pytest.raises(ValueError, match="missing modelfit results"):
        validate_input(model=model)


def test_validate_input_raises_cwres(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = remove_covariance_step(model)
    modelfit_results = replace(res, residuals=res.residuals.drop(columns=['CWRES']))
    model = model.replace(modelfit_results=modelfit_results)

    with pytest.raises(ValueError, match="CWRES"):
        validate_input(model=model)


def test_validate_input_raises_cipredi(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    model = remove_covariance_step(model)
    modelfit_results = replace(res, predictions=res.predictions.drop(columns=['CIPREDI']))
    model = model.replace(modelfit_results=modelfit_results)

    with pytest.raises(ValueError, match="IPRED"):
        validate_input(model=model)


def test_validate_input_raises_ipred(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    model = remove_covariance_step(model)
    modelfit_results = replace(res, predictions=res.predictions.drop(columns=['IPRED']))
    model = model.replace(modelfit_results=modelfit_results)

    with pytest.raises(ValueError, match="IPRED"):
        validate_input(model=model)
