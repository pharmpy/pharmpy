import numpy as np
import pandas as pd
import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import NormalDistribution
from pharmpy.modeling import set_seq_zo_fo_absorption
from pharmpy.tools import fit, retrieve_models, run_iivsearch


def test_no_of_etas_keep(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res_keep1 = run_iivsearch(
            'brute_force_no_of_etas',
            results=start_model.modelfit_results,
            model=start_model,
            keep=["CL"],
        )
        no_of_models = 8
        assert len(res_keep1.summary_models) == no_of_models // 2
        assert res_keep1.summary_individuals.iloc[-1]['description'] == '[CL]'


def test_block_structure(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_iivsearch(
            'brute_force_block_structure', results=start_model.modelfit_results, model=start_model
        )

        no_of_candidate_models = 4
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models

        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        assert all(model.random_variables != start_model.random_variables for model in res.models)

        assert res.summary_tool.loc[1, 'mox2']['description'] == '[CL]+[VC]+[MAT]'
        input_model = retrieve_models(res, names=['input_model'])[0]
        assert isinstance(input_model.random_variables['ETA_1'], NormalDistribution)

        assert res.summary_tool.loc[1, 'iivsearch_run1']['description'] == '[CL,VC,MAT]'
        assert len(res.models[0].random_variables['ETA_1'].names) == 3

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


def test_no_of_etas(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_iivsearch(
            'brute_force_no_of_etas', results=start_model.modelfit_results, model=start_model
        )

        no_of_candidate_models = 7
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models

        assert res.models[-1].modelfit_results

        assert res.summary_tool.loc[1, 'mox2']['description'] == '[CL]+[VC]+[MAT]'
        input_model = retrieve_models(res, names=['input_model'])[0]
        assert input_model.random_variables.iiv.names == ['ETA_1', 'ETA_2', 'ETA_3']

        assert res.summary_tool.iloc[-1]['description'] == '[]'
        assert res.models[0].random_variables.iiv.names == ['ETA_2', 'ETA_3']

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


def test_brute_force(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_iivsearch('brute_force', results=start_model.modelfit_results, model=start_model)

        no_of_candidate_models = 8
        assert len(res.summary_tool) == no_of_candidate_models + 2
        assert len(res.summary_models) == no_of_candidate_models + 1
        assert len(res.models) == no_of_candidate_models

        assert 'iivsearch_run3' in res.summary_errors.index.get_level_values('model')

        assert all(
            model.modelfit_results and not np.isnan(model.modelfit_results.ofv)
            for model in res.models
        )
        assert all(model.random_variables != start_model.random_variables for model in res.models)

        summary_tool_sorted_by_dbic_step1 = res.summary_tool.loc[1].sort_values(
            by=['dbic'], ascending=False
        )
        summary_tool_sorted_by_bic_step1 = res.summary_tool.loc[1].sort_values(by=['bic'])
        summary_tool_sorted_by_rank_step1 = res.summary_tool.loc[1].sort_values(by=['rank'])
        pd.testing.assert_frame_equal(
            summary_tool_sorted_by_dbic_step1, summary_tool_sorted_by_rank_step1
        )
        pd.testing.assert_frame_equal(
            summary_tool_sorted_by_dbic_step1, summary_tool_sorted_by_bic_step1
        )

        summary_tool_sorted_by_dbic_step2 = res.summary_tool.loc[2].sort_values(
            by=['dbic'], ascending=False
        )
        summary_tool_sorted_by_bic_step2 = res.summary_tool.loc[2].sort_values(by=['bic'])
        summary_tool_sorted_by_rank_step2 = res.summary_tool.loc[2].sort_values(by=['rank'])
        pd.testing.assert_frame_equal(
            summary_tool_sorted_by_dbic_step2, summary_tool_sorted_by_rank_step2
        )
        pd.testing.assert_frame_equal(
            summary_tool_sorted_by_dbic_step2, summary_tool_sorted_by_bic_step2
        )

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    'iiv_strategy',
    ['add_diagonal', 'fullblock'],
)
def test_no_of_etas_iiv_strategies(tmp_path, model_count, start_model, iiv_strategy):
    with chdir(tmp_path):
        start_model = start_model.replace(name='moxo2_copy', modelfit_results=None)
        start_model = set_seq_zo_fo_absorption(start_model)
        start_res = fit(start_model)
        start_model = start_model.replace(modelfit_results=start_res)

        res = run_iivsearch(
            'brute_force_no_of_etas',
            iiv_strategy=iiv_strategy,
            results=start_model.modelfit_results,
            model=start_model,
        )

        if iiv_strategy == 'fullblock':
            base_model = [model for model in res.models if model.name == 'base_model'].pop()
            base_rvs = base_model.random_variables.iiv
            assert len(base_rvs['ETA_1']) == base_rvs.nrvs

        no_of_candidate_models = 15
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 2
        assert len(res.models) == no_of_candidate_models + 1
        assert res.models[-1].modelfit_results

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models + 1
        assert (rundir / 'metadata.json').exists()
