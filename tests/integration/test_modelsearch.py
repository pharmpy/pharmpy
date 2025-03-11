import pytest

from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_modelsearch


def test_modelsearch_nonmem(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_modelsearch(
            model=start_modelres[0],
            results=start_modelres[1],
            search_space='ABSORPTION([FO,ZO]);PERIPHERALS([0,1])',
            algorithm='exhaustive',
            rank_type='mbic',
            E=1.0,
        )

        assert len(res.summary_tool) == 4
        assert len(res.summary_models) == 4
        assert len(res.models) == 4

        rundir = tmp_path / 'modelsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == 5
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()


@pytest.mark.parametrize(
    'search_space, algorithm, kwargs, has_base_model, no_of_cands, max_added_params, best_model',
    [
        (
            'ABSORPTION([FO,ZO]);PERIPHERALS([0,1])',
            'exhaustive',
            dict(),
            False,
            3,
            2,
            'modelsearch_run2',
        ),
        (
            'ABSORPTION([FO,ZO]);PERIPHERALS([0,1])',
            'exhaustive_stepwise',
            dict(),
            False,
            4,
            2,
            'modelsearch_run4',
        ),
        (
            'ABSORPTION([FO,ZO,SEQ-ZO-FO]);PERIPHERALS([0,1])',
            'exhaustive_stepwise',
            dict(),
            False,
            7,
            4,
            'modelsearch_run5',
        ),
        (
            'ABSORPTION([FO,ZO]);LAGTIME([OFF,ON])',
            'exhaustive_stepwise',
            {'iiv_strategy': 'no_add'},
            False,
            4,
            1,
            'modelsearch_run4',
        ),
        (
            'ABSORPTION([FO,ZO]);LAGTIME([OFF,ON])',
            'exhaustive_stepwise',
            {'iiv_strategy': 'add_diagonal'},
            False,
            4,
            2,
            'modelsearch_run4',
        ),
        (
            'ABSORPTION([FO,ZO]);LAGTIME([OFF,ON])',
            'exhaustive_stepwise',
            {'iiv_strategy': 'fullblock'},
            False,
            4,
            8,
            'modelsearch_run4',
        ),
        (
            'ABSORPTION(FO);PERIPHERALS(0..2);ALLOMETRY(WT,70)',
            'exhaustive_stepwise',
            dict(),
            True,
            3,
            4,
            'modelsearch_run2',
        ),
    ],
)
def test_modelsearch_dummy(
    tmp_path,
    model_count,
    start_modelres_dummy,
    search_space,
    algorithm,
    kwargs,
    has_base_model,
    no_of_cands,
    max_added_params,
    best_model,
):
    with chdir(tmp_path):
        res = run_modelsearch(
            model=start_modelres_dummy[0],
            results=start_modelres_dummy[1],
            search_space=search_space,
            algorithm=algorithm,
            esttool='dummy',
            **kwargs
        )

        if has_base_model:
            no_of_ranked_models = no_of_cands
        else:
            no_of_ranked_models = no_of_cands + 1

        assert len(res.summary_tool) == no_of_ranked_models
        assert len(res.summary_models) == no_of_cands + 1
        assert len(res.models) == no_of_ranked_models
        assert res.summary_tool['d_params'].max() == max_added_params

        # FIXME: move to unit test
        summary_tool_sorted_by_dbic = res.summary_tool.sort_values(by=['dbic'], ascending=False)
        summary_tool_sorted_by_bic = res.summary_tool.sort_values(by=['bic'])
        summary_tool_sorted_by_rank = res.summary_tool.sort_values(by=['rank'])
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_rank)
        pd.testing.assert_frame_equal(summary_tool_sorted_by_dbic, summary_tool_sorted_by_bic)

        assert res.final_model.name == best_model

        rundir = tmp_path / 'modelsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_cands + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
        assert (rundir / 'models' / 'modelsearch_run1' / 'model_results.json').exists()
        assert not (rundir / 'models' / 'modelsearch_run1' / 'model.lst').exists()
