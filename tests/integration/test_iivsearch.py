import pytest

from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import NormalDistribution
from pharmpy.modeling import set_seq_zo_fo_absorption
from pharmpy.tools import fit, run_iivsearch
from pharmpy.workflows import LocalDirectoryContext

# FIXME: Tests including modelfit_results are commented out, uncomment once function retrieve_models
#  return model entries or we have a separate function for this


@pytest.mark.parametrize(
    ('algorithm', 'keep', 'no_of_candidate_models'),
    (('top_down_exhaustive', ['CL'], 8), ('bottom_up_stepwise', ["VC"], 10)),
)
def test_no_of_etas_keep(
    tmp_path, algorithm, keep, no_of_candidate_models, model_count, start_modelres
):
    with chdir(tmp_path):
        res_keep1 = run_iivsearch(
            algorithm,
            results=start_modelres[1],
            model=start_modelres[0],
            keep=keep,
            correlation_algorithm="skip",
        )
        no_of_models = no_of_candidate_models
        assert len(res_keep1.summary_models) == no_of_models // 2
        if algorithm == "top_down_exhaustive":
            assert res_keep1.summary_individuals.iloc[-1]['description'] == '[CL]'
        elif algorithm == "bottom_up_stepwise":
            assert res_keep1.summary_models.iloc[1]['description'] == '[VC]'


def test_block_structure(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_iivsearch(
            'skip',
            results=start_modelres[1],
            model=start_modelres[0],
            correlation_algorithm="top_down_exhaustive",
        )

        no_of_candidate_models = 4
        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1

        ctx = LocalDirectoryContext("iivsearch1")
        names = ctx.list_all_names()
        res_models = [
            ctx.retrieve_model_entry(name).model for name in names if name not in ['input', 'final']
        ]
        assert len(res_models) == no_of_candidate_models

        start_model = start_modelres[0]
        assert all(model.random_variables != start_model.random_variables for model in res_models)

        assert res.summary_tool.loc[1, 'mox2']['description'] == '[CL]+[VC]+[MAT]'
        assert isinstance(start_model.random_variables['ETA_1'], NormalDistribution)

        assert res.summary_tool.loc[1, 'iivsearch_run1']['description'] == '[CL,VC,MAT]'
        assert len(res_models[0].random_variables['ETA_1'].names) == 3

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models + 2
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    ('algorithm', 'correlation_algorithm', 'no_of_candidate_models'),
    (('top_down_exhaustive', 'skip', 7),),  # ('bottom_up_stepwise', 'skip', 4)
)
def test_no_of_etas_base(
    tmp_path, model_count, start_modelres, algorithm, correlation_algorithm, no_of_candidate_models
):
    with chdir(tmp_path):
        res = run_iivsearch(
            algorithm,
            results=start_modelres[1],
            model=start_modelres[0],
            keep=[],
            correlation_algorithm=correlation_algorithm,
        )

        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 1

        ctx = LocalDirectoryContext('iivsearch1')
        names = ctx.list_all_names()
        res_models = [
            ctx.retrieve_model_entry(name).model for name in names if name not in ['input', 'final']
        ]
        assert len(res_models) == no_of_candidate_models

        assert res.summary_tool.loc[1, 'mox2']['description'] == '[CL]+[VC]+[MAT]'
        assert start_modelres[0].random_variables.iiv.names == ['ETA_1', 'ETA_2', 'ETA_3']

        if algorithm == 'top_down_exhaustive':
            assert res.summary_tool.iloc[-1]['description'] == '[]'
            assert res_models[0].random_variables.iiv.names == ['ETA_2', 'ETA_3']
        elif algorithm == 'bottom_up_stepwise':
            assert res.summary_tool.iloc[-1]['description'] == '[CL]'
            assert res_models[0].random_variables.iiv.names == ['ETA_1']

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models + 2
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    ('algorithm', 'no_of_candidate_models'),
    (('top_down_exhaustive', 8), ('bottom_up_stepwise', 8)),
)
def test_brute_force(tmp_path, model_count, start_modelres, algorithm, no_of_candidate_models):
    with chdir(tmp_path):
        res = run_iivsearch(algorithm, keep=[], results=start_modelres[1], model=start_modelres[0])

        assert len(res.summary_tool) == no_of_candidate_models + 2
        assert len(res.summary_models) == no_of_candidate_models + 1

        ctx = LocalDirectoryContext('iivsearch1')
        names = ctx.list_all_names()
        res_models = [
            ctx.retrieve_model_entry(name).model for name in names if name not in ['input', 'final']
        ]
        assert len(res_models) == no_of_candidate_models

        if algorithm == 'top_down_exhaustive':
            assert 'iivsearch_run3' in res.summary_errors.index.get_level_values('model')

            assert all(
                model.random_variables != start_modelres[0].random_variables for model in res_models
            )
        elif algorithm == 'bottom_up_stepwise':
            cand_vs_input = [
                model.random_variables == start_modelres[0].random_variables for model in res_models
            ]
            assert len([c for c in cand_vs_input if c]) == 1

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

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models + 2
        assert (rundir / 'metadata.json').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ('algorithm', 'correlation_algorithm', 'iiv_strategy', 'no_of_candidate_models'),
    (
        ('top_down_exhaustive', 'skip', 'add_diagonal', 15),
        ('top_down_exhaustive', 'skip', 'fullblock', 15),
    ),
)
def test_no_of_etas_iiv_strategies(
    tmp_path,
    model_count,
    start_modelres,
    algorithm,
    correlation_algorithm,
    iiv_strategy,
    no_of_candidate_models,
):
    with chdir(tmp_path):
        start_model = start_modelres[0].replace(name='moxo2_copy')
        start_model = set_seq_zo_fo_absorption(start_model)
        start_res = fit(start_model)

        res = run_iivsearch(
            algorithm,
            iiv_strategy=iiv_strategy,
            results=start_res,
            model=start_model,
            keep=[],
            correlation_algorithm=correlation_algorithm,
        )

        assert len(res.summary_tool) == no_of_candidate_models + 1
        assert len(res.summary_models) == no_of_candidate_models + 2

        ctx = LocalDirectoryContext('iivsearch1')
        names = ctx.list_all_names()
        res_models = [
            ctx.retrieve_model_entry(name).model for name in names if name not in ['input', 'final']
        ]
        assert len(res_models) == no_of_candidate_models + 1

        if iiv_strategy == 'fullblock':
            base_model = [model for model in res_models if model.name == 'base_model'].pop()
            base_rvs = base_model.random_variables.iiv
            assert len(base_rvs['ETA_1']) == base_rvs.nrvs

        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidate_models + 3
        assert (rundir / 'metadata.json').exists()
