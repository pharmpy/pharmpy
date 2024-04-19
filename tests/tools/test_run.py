import inspect

# import os
import shutil
from pathlib import Path
from typing import get_type_hints

import pytest

import pharmpy
from pharmpy.deps import numpy as np
from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_iiv, load_example_model, read_model, set_lower_bounds
from pharmpy.tools.run import (  # retrieve_final_model,; retrieve_models,
    _create_metadata_common,
    _create_metadata_tool,
    import_tool,
    is_strictness_fulfilled,
    load_example_modelfit_results,
    rank_models,
    read_modelfit_results,
    summarize_errors_from_entries,
    summarize_modelfit_results_from_entries,
)
from pharmpy.workflows import LocalDirectoryContext, ModelEntry, local_dask


@pytest.mark.parametrize(
    ('args', 'kwargs'),
    (
        (
            ('ABSORPTION(ZO)', 'exhaustive'),
            {'iiv_strategy': 'no_add'},
        ),
        (
            ('ABSORPTION(ZO)',),
            {'algorithm': 'exhaustive'},
        ),
    ),
)
def test_create_metadata_tool(tmp_path, pheno, args, kwargs):
    with chdir(tmp_path):
        tool_name = 'modelsearch'
        database = LocalDirectoryContext(tool_name)
        tool = import_tool(tool_name)
        tool_params = inspect.signature(tool.create_workflow).parameters
        tool_param_types = get_type_hints(tool.create_workflow)

        metadata = _create_metadata_tool(
            database=database,
            tool_name=tool_name,
            tool_params=tool_params,
            tool_param_types=tool_param_types,
            args=args,
            kwargs={'model': pheno, **kwargs},
        )

        rundir = tmp_path / 'modelsearch'

        assert (rundir / 'models').exists()

        assert metadata['pharmpy_version'] == pharmpy.__version__
        assert metadata['tool_name'] == 'modelsearch'
        assert metadata['tool_options']['model']['__class__'] == 'Model'
        assert metadata['tool_options']['model']['arg_name'] == 'pheno_real'
        assert metadata['tool_options']['model']['db_name'] == 'input_model'
        assert metadata['tool_options']['rank_type'] == 'mbic'
        assert metadata['tool_options']['algorithm'] == 'exhaustive'


def test_create_metadata_tool_raises(tmp_path, pheno):
    with chdir(tmp_path):
        tool_name = 'modelsearch'
        database = LocalDirectoryContext(tool_name)
        tool = import_tool(tool_name)
        tool_params = inspect.signature(tool.create_workflow).parameters
        tool_param_types = get_type_hints(tool.create_workflow)
        with pytest.raises(Exception, match='modelsearch: \'algorithm\' was not set'):
            _create_metadata_tool(
                database=database,
                tool_name=tool_name,
                tool_params=tool_params,
                tool_param_types=tool_param_types,
                args=('ABSORPTION(ZO)',),
                kwargs={'model': pheno},
            )


def test_create_metadata_common(tmp_path):
    with chdir(tmp_path):
        name = 'modelsearch'

        dispatcher = local_dask
        database = LocalDirectoryContext(name, Path.cwd())

        metadata = _create_metadata_common(
            database=database,
            dispatcher=dispatcher,
            toolname=name,
            common_options={},
        )

        assert metadata['dispatcher'] == 'pharmpy.workflows.dispatchers.local_dask'
        assert metadata['context']['class'] == 'LocalDirectoryContext'
        path = Path(metadata['context']['path'])
        assert path.stem == 'modelsearch'
        assert 'path' not in metadata.keys()

        path = 'tool_database_path'

        dispatcher = local_dask
        database = LocalDirectoryContext(path)

        metadata = _create_metadata_common(
            database=database,
            dispatcher=dispatcher,
            toolname=name,
            common_options={'path': path},
        )

        path = Path(metadata['context']['path'])
        assert path.stem == 'tool_database_path'
        assert metadata['path'] == 'tool_database_path'


def test_summarize_errors(load_model_for_test, testdata, tmp_path, pheno_path):
    with chdir(tmp_path):
        model = read_model(pheno_path)
        res = read_modelfit_results(pheno_path)
        me1 = ModelEntry(model=model, modelfit_results=res)
        shutil.copy2(testdata / 'pheno_data.csv', tmp_path)

        error_path = testdata / 'nonmem' / 'errors'

        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_no_header.mod')
        shutil.copy2(error_path / 'no_header_error.lst', tmp_path / 'pheno_no_header.lst')
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.ext', tmp_path / 'pheno_no_header.ext')
        model_no_header = read_model('pheno_no_header.mod')
        res_no_header = read_modelfit_results('pheno_no_header.mod')
        me2 = ModelEntry(model=model_no_header, modelfit_results=res_no_header)

        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_rounding_error.mod')
        shutil.copy2(error_path / 'rounding_error.lst', tmp_path / 'pheno_rounding_error.lst')
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.ext', tmp_path / 'pheno_rounding_error.ext')
        model_rounding_error = read_model('pheno_rounding_error.mod')
        res_rounding_error = read_modelfit_results('pheno_rounding_error.mod')
        me3 = ModelEntry(model=model_rounding_error, modelfit_results=res_rounding_error)

        entries = [me1, me2, me3]
        summary = summarize_errors_from_entries(entries)

        assert 'pheno_real' not in summary.index.get_level_values('model')
        assert len(summary.loc[('pheno_no_header', 'WARNING')]) == 1
        assert len(summary.loc[('pheno_no_header', 'ERROR')]) == 2
        assert len(summary.loc[('pheno_rounding_error', 'ERROR')]) == 2


class DummyModel:
    def __init__(self, name, parameter_names):
        self.name = name
        self.parameters = parameter_names


class DummyResults:
    def __init__(
        self,
        name,
        ofv,
        minimization_successful=True,
        termination_cause=None,
        significant_digits=5,
        warnings=[],
    ):
        self.name = name
        self.ofv = ofv
        self.minimization_successful = minimization_successful
        self.termination_cause = termination_cause
        # 5 is an arbitrary number, this is relevant in test if sig. digits is unreportable (NaN)
        self.significant_digits = significant_digits
        self.warnings = warnings


def test_rank_models():
    base = DummyModel('base', parameter_names=['p1'])
    m1 = DummyModel('m1', parameter_names=['p1', 'p2'])
    m2 = DummyModel('m2', parameter_names=['p1', 'p2'])
    m3 = DummyModel('m3', parameter_names=['p1', 'p2', 'p3'])
    m4 = DummyModel('m4', parameter_names=['p1'])

    base_res = DummyResults(name=base.name, ofv=0)
    m1_res = DummyResults(
        name=m1.name, ofv=-5, minimization_successful=False, termination_cause='rounding_errors'
    )
    m2_res = DummyResults(name=m2.name, ofv=-4)
    m3_res = DummyResults(name=m3.name, ofv=-4)
    m4_res = DummyResults(name=m4.name, ofv=1)

    models = [m1, m2, m3, m4]
    models_res = [m1_res, m2_res, m3_res, m4_res]

    df = rank_models(base, base_res, models, models_res, rank_type='ofv')
    assert len(df) == 5
    best_model = df.loc[df['rank'] == 1].index.values
    assert list(best_model) == ['m2', 'm3']

    # Test if rounding errors are allowed
    df = rank_models(
        base,
        base_res,
        models,
        models_res,
        strictness='minimization_successful or (rounding_errors and sigdigs>=0)',
        rank_type='ofv',
    )
    best_model = df.loc[df['rank'] == 1].index.values
    assert list(best_model) == ['m1']
    ranked_models = df.dropna().index.values
    assert len(ranked_models) == 5

    # Test with a cutoff of dOFV=1
    df = rank_models(base, base_res, models, models_res, rank_type='ofv', cutoff=1)
    ranked_models = df.dropna().index.values
    assert len(ranked_models) == 3

    # Test with LRT
    df = rank_models(base, base_res, models, models_res, rank_type='lrt', cutoff=0.05)
    ranked_models = list(df.dropna().index.values)
    assert sorted(ranked_models) == ['base', 'm2']

    df = rank_models(
        base,
        base_res,
        models,
        models_res,
        parent_dict={m: base for m in models},
        rank_type='lrt',
        cutoff=0.05,
    )
    ranked_models = list(df.dropna().index.values)
    assert sorted(ranked_models) == ['base', 'm2']

    # Test if candidate model does not have an OFV
    m5 = DummyModel('m5', parameter_names=['p1'])
    m5_res = DummyResults(name=m5.name, ofv=np.nan)
    df = rank_models(base, base_res, models + [m5], models_res + [m5_res], rank_type='ofv')
    ranked_models = list(df.dropna().index.values)
    assert 'm5' not in ranked_models
    assert np.isnan(df.loc['m5']['rank'])

    # Test if model has minimized but has unreportable number of significant digits while still allowing rounding
    # errors
    m6 = DummyModel('m6', parameter_names=['p1'])
    m6_res = DummyResults(
        name=m6.name,
        ofv=-5,
        minimization_successful=False,
        termination_cause='rounding_errors',
        significant_digits=np.nan,
    )
    df = rank_models(
        base,
        base_res,
        models + [m6],
        models_res + [m6_res],
        strictness='minimization_successful or (rounding_errors and sigdigs>=0)',
        rank_type='ofv',
    )
    ranked_models = list(df.dropna().index.values)
    assert 'm6' not in ranked_models
    assert np.isnan(df.loc['m6']['rank'])

    # Test if base model failed, fall back to rank value
    base_nan = DummyModel('base_nan', parameter_names=['p1'])
    base_nan_res = DummyResults(name=base_nan.name, ofv=np.nan)
    df = rank_models(
        base_nan,
        base_nan_res,
        models,
        models_res,
        strictness='minimization_successful or (rounding_errors and sigdigs>=0)',
        rank_type='ofv',
    )
    assert df.iloc[0].name == 'm1'

    # Test if base model failed but still has OFV with a very high value, fall back to rank value
    base_nan = DummyModel(
        'base_nan',
        parameter_names=['p1'],
    )
    base_nan_res = DummyResults(
        name=base_nan.name, ofv=2e154, minimization_successful=False, termination_cause='something'
    )
    df = rank_models(
        base_nan,
        base_nan_res,
        models,
        models_res,
        strictness='minimization_successful or (rounding_errors and sigdigs>=0)',
        rank_type='ofv',
    )
    assert df.iloc[0].name == 'm1'
    assert df.nunique()['ofv'] == df.nunique()['rank']

    with pytest.raises(ValueError):
        rank_models(base, base_res, models + [m5], models_res)


def test_rank_models_bic(load_model_for_test, testdata):
    model_base = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    model_iiv = add_iiv(model_base, ['S1'], 'exp')
    model_iiv = model_iiv.replace(name='pheno_iiv')
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')
    df_mixed = rank_models(model_base, res, [model_iiv], [res], rank_type='bic', bic_type='mixed')
    df_mult_test = rank_models(
        model_base,
        res,
        [model_iiv],
        [res],
        rank_type='bic',
        bic_type='mixed',
        multiple_testing=True,
        mult_test_p=2,
    )
    assert df_mixed.loc['pheno_iiv', 'bic'] != df_mult_test.loc['pheno_iiv', 'bic']


def test_summarize_modelfit_results(
    load_model_for_test, create_model_for_test, testdata, pheno_path
):
    pheno = read_model(pheno_path)
    pheno_res = read_modelfit_results(pheno_path)
    pheno_me = ModelEntry(model=pheno, modelfit_results=pheno_res)

    summary_single = summarize_modelfit_results_from_entries([pheno_me])

    assert summary_single.loc['pheno_real']['ofv'] == 586.2760562818805
    assert summary_single['IVCL_estimate'].mean() == 0.0293508

    assert len(summary_single.index) == 1

    mox = read_model(testdata / 'nonmem' / 'models' / 'mox1.mod')
    mox_res = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox1.mod')
    mox_me = ModelEntry(model=mox, modelfit_results=mox_res)

    summary_multiple = summarize_modelfit_results_from_entries([pheno_me, mox_me])

    assert summary_multiple.loc['mox1']['ofv'] == -624.5229577248352
    assert summary_multiple['IIV_CL_estimate'].mean() == 0.41791
    assert summary_multiple['IIV_CL_V_estimate'].mean() == 0.395647  # One is NaN

    assert len(summary_multiple.index) == 2
    assert list(summary_multiple.index) == ['pheno_real', 'mox1']

    summary_no_res = summarize_modelfit_results_from_entries([pheno_me, None])

    assert summary_no_res.loc['pheno_real']['ofv'] == 586.2760562818805

    multest_path = (
        testdata
        / 'nonmem'
        / 'modelfit_results'
        / 'onePROB'
        / 'multEST'
        / 'noSIM'
        / 'pheno_multEST.mod'
    )
    pheno_multest = read_model(multest_path)
    pheno_multest_res = read_modelfit_results(multest_path)
    pheno_multest_me = ModelEntry(model=pheno_multest, modelfit_results=pheno_multest_res)

    summary_multest = summarize_modelfit_results_from_entries([pheno_multest_me, mox_me])

    assert len(summary_multest.index) == 2

    assert not summary_multest.loc['pheno_multEST']['minimization_successful']
    summary_multest_full = summarize_modelfit_results_from_entries(
        [pheno_multest_me, mox_me], include_all_execution_steps=True
    )

    assert len(summary_multest_full.index) == 3
    assert len(set(summary_multest_full.index.get_level_values('model'))) == 2
    assert summary_multest_full.loc['pheno_multEST', 1]['run_type'] == 'estimation'
    assert summary_multest_full.loc['pheno_multEST', 2]['run_type'] == 'evaluation'

    assert not summary_multest_full.loc['pheno_multEST', 1]['minimization_successful']

    summary_multest_full_no_res = summarize_modelfit_results_from_entries(
        [None, mox_me],
        include_all_execution_steps=True,
    )

    assert summary_multest_full_no_res.loc['mox1', 1]['ofv'] == -624.5229577248352

    with pytest.raises(ValueError, match='Option `results` is None'):
        summarize_modelfit_results_from_entries(None)

    with pytest.raises(ValueError, match='All input results are empty'):
        summarize_modelfit_results_from_entries([None, None])


def test_summarize_modelfit_results_errors(load_model_for_test, testdata, tmp_path, pheno_path):
    with chdir(tmp_path):
        model = read_model(pheno_path)
        res = read_modelfit_results(pheno_path)
        me1 = ModelEntry(model=model, modelfit_results=res)
        shutil.copy2(testdata / 'pheno_data.csv', tmp_path)

        error_path = testdata / 'nonmem' / 'errors'

        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_no_header.mod')
        shutil.copy2(error_path / 'no_header_error.lst', tmp_path / 'pheno_no_header.lst')
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.ext', tmp_path / 'pheno_no_header.ext')
        model_no_header = read_model('pheno_no_header.mod')
        model_no_header_res = read_modelfit_results('pheno_no_header.mod')
        me2 = ModelEntry(model=model_no_header, modelfit_results=model_no_header_res)

        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_rounding_error.mod')
        shutil.copy2(error_path / 'rounding_error.lst', tmp_path / 'pheno_rounding_error.lst')
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.ext', tmp_path / 'pheno_rounding_error.ext')
        model_rounding_error = read_model('pheno_rounding_error.mod')
        model_rounding_error_res = read_modelfit_results('pheno_rounding_error.mod')
        me3 = ModelEntry(model=model_rounding_error, modelfit_results=model_rounding_error_res)

        entries = [
            me1,
            me2,
            me3,
        ]
        summary = summarize_modelfit_results_from_entries(entries)

        assert summary.loc['pheno_real']['errors_found'] == 0
        assert summary.loc['pheno_real']['warnings_found'] == 0
        assert summary.loc['pheno_no_header']['errors_found'] == 2
        assert summary.loc['pheno_no_header']['warnings_found'] == 1
        assert summary.loc['pheno_rounding_error']['errors_found'] == 2
        assert summary.loc['pheno_rounding_error']['warnings_found'] == 0


def test_read_modelfit_results(testdata):
    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_real.mod')
    assert res.ofv == 586.27605628188053
    expected_rse = {
        'PTVCL': 0.04473086219931638,
        'PTVV': 0.027325355750219965,
        'THETA_3': 0.5270721117543418,
        'IVCL': 0.4570676097414721,
        'IVV': 0.2679176521178241,
        'SIGMA_1_1': 0.17214711879767391,
    }
    assert res.relative_standard_errors.to_dict() == expected_rse

    res = read_modelfit_results(testdata / 'nonmem' / 'pheno_design.mod')
    assert res.ofv == 730.9699060285753
    expected_rse = {
        'TVCL': 0.07540090004436839,
        'TVV': 0.06030486531634996,
        'IVCL': 0.3895984361000244,
        'IVV': 0.19632785003247685,
        'SIGMA_1_1': 0.1865063481541913,
    }
    assert res.relative_standard_errors.to_dict() == expected_rse


def test_load_example_modelfit_results():
    res = load_example_modelfit_results("pheno")
    assert res.ofv == 586.27605628188053


@pytest.mark.parametrize(
    'path, statement, expected',
    [
        (
            'nonmem/modelfit_results/onePROB/oneEST/noSIM/near_bounds.mod',
            'minimization_successful or (rounding_errors and sigdigs > 0)',
            True,
        ),
        (
            'nonmem/pheno_real.mod',
            'condition_number < 1000',
            False,
        ),
        (
            'nonmem/pheno_real.mod',
            'condition_number < 300000',
            True,
        ),
        (
            'nonmem/modelfit_results/onePROB/oneEST/noSIM/maxeval3.mod',
            'rse < 30',
            True,
        ),
        (
            'nonmem/pheno.mod',
            'minimization_successful and sigdigs > 0 and rse<2',
            False,
        ),
        (
            'nonmem/pheno.mod',
            'minimization_successful and sigdigs > 3',
            True,
        ),
        (
            'nonmem/pheno.mod',
            'final_zero_gradient',
            False,
        ),
        (
            'nonmem/pheno.mod',
            'final_zero_gradient_theta',
            False,
        ),
        (
            'nonmem/pheno.mod',
            'final_zero_gradient_omega',
            False,
        ),
        (
            'nonmem/pheno.mod',
            'final_zero_gradient_sigma',
            False,
        ),
    ],
)
def test_strictness(testdata, path, statement, expected):
    res = read_modelfit_results(testdata / path)
    model = read_model(testdata / path)
    assert is_strictness_fulfilled(res, model, statement) == expected


def test_strictness_unallowed_operators(testdata):
    res = read_modelfit_results(testdata / 'nonmem/pheno.mod')
    model = read_model(testdata / 'nonmem/pheno.mod')
    with pytest.raises(ValueError, match=r"Unallowed operators found: &"):
        is_strictness_fulfilled(res, model, 'minimization_successful & rounding_errors')
    with pytest.raises(ValueError, match=r"Unallowed operators found: &, |"):
        is_strictness_fulfilled(
            res, model, 'minimization_successful & (rounding_errors | sigdigs>3)'
        )


def test_strictness_parameters(testdata):
    res = load_example_modelfit_results('pheno')
    model = load_example_model("pheno")
    assert not is_strictness_fulfilled(res, model, 'rse_theta < 0.3')
    assert is_strictness_fulfilled(res, model, 'rse_theta < 0.55')
    assert not is_strictness_fulfilled(res, model, 'rse_omega < 0.3')
    assert is_strictness_fulfilled(res, model, 'rse_omega < 0.5')
    assert is_strictness_fulfilled(res, model, 'rse_sigma < 0.2')

    res = read_modelfit_results(testdata / 'nonmem/pheno.mod')
    model = read_model(testdata / 'nonmem/pheno.mod')
    assert not is_strictness_fulfilled(res, model, 'estimate_near_boundary_theta')
    model = set_lower_bounds(model, {'TVCL': 0.0058})
    assert is_strictness_fulfilled(res, model, 'estimate_near_boundary_theta')
