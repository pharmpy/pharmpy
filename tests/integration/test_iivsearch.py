import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_lag_time
from pharmpy.tools import fit, run_iivsearch
from pharmpy.workflows import LocalDirectoryContext


@pytest.mark.parametrize(
    'algorithm, correlation_algorithm, kwargs, no_of_candidate_models, max_diff_params, best_model',
    [
        ('top_down_exhaustive', 'skip', {'keep': []}, 7, 3, 'iivsearch_run1'),
        ('bottom_up_stepwise', 'skip', {'keep': []}, 4, 2, 'iivsearch_run2'),
        ('skip', 'top_down_exhaustive', dict(), 4, 3, 'iivsearch_run1'),
        ('top_down_exhaustive', 'skip', dict(), 3, 2, 'iivsearch_run1'),
        ('top_down_exhaustive', None, {'keep': []}, 8, 3, 'iivsearch_run8'),
        ('bottom_up_stepwise', None, {'keep': []}, 5, 2, 'iivsearch_run5'),
        ('top_down_exhaustive', 'skip', {'iiv_strategy': 'add_diagonal'}, 7, 3, 'iivsearch_run1'),
        ('top_down_exhaustive', 'skip', {'iiv_strategy': 'fullblock'}, 7, 9, 'iivsearch_run7'),
        ('bottom_up_stepwise', 'skip', {'iiv_strategy': 'fullblock'}, 7, 9, 'iivsearch_run7'),
        (
            'top_down_exhaustive',
            None,
            {'rank_type': 'mbic', 'E_p': '50%', 'E_q': '50%'},
            4,
            2,
            'iivsearch_run4',
        ),
        (
            'bottom_up_stepwise',
            None,
            {'rank_type': 'mbic', 'E_p': '50%', 'E_q': '50%'},
            5,
            2,
            'iivsearch_run5',
        ),
    ],
)
def test_iivsearch_dummy(
    tmp_path,
    model_count,
    start_modelres_dummy,
    algorithm,
    correlation_algorithm,
    kwargs,
    no_of_candidate_models,
    max_diff_params,
    best_model,
):
    with chdir(tmp_path):
        no_of_models_total = no_of_candidate_models + 1  # Include input
        has_iiv_strategy = 'iiv_strategy' in kwargs and kwargs['iiv_strategy'] != 'no_add'
        if has_iiv_strategy:
            start_model = add_lag_time(start_modelres_dummy[0])
            start_res = fit(start_model, esttool='dummy')
            if algorithm != 'bottom_up_stepwise':
                no_of_models_total += 1  # Include base
        else:
            start_model = start_modelres_dummy[0]
            start_res = start_modelres_dummy[1]

        res = run_iivsearch(
            model=start_model,
            results=start_res,
            algorithm=algorithm,
            correlation_algorithm=correlation_algorithm,
            esttool='dummy',
            **kwargs
        )

        algorithms = list(filter(lambda x: x != 'skip', [algorithm, correlation_algorithm]))
        assert len(res.summary_tool.index.get_level_values('step').unique()) == len(algorithms) + 1
        assert len(res.summary_tool.index.get_level_values('model').unique()) == no_of_models_total
        assert len(res.summary_models) == no_of_models_total
        assert res.summary_tool['d_params'].abs().max() == max_diff_params

        assert res.final_model.name == best_model

        ctx = LocalDirectoryContext('iivsearch1')
        cand_model = ctx.retrieve_model_entry('iivsearch_run2').model
        assert cand_model.random_variables != start_model.random_variables

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_models_total + 1  # Include final
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
        assert (rundir / 'models' / 'iivsearch_run1' / 'model_results.json').exists()
        assert not (rundir / 'models' / 'iivsearch_run1' / 'model.lst').exists()


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
@pytest.mark.parametrize(
    ('algorithm', 'correlation_algorithm', 'no_of_candidate_models', 'strategy'),
    (
        ('top_down_exhaustive', 'skip', 3, 'fullblock'),
        ('bottom_up_stepwise', 'skip', 4, 'no_add'),
        # ('bottom_up_stepwise', 'skip', 4, 'fullblock'),
        ('bottom_up_stepwise', 'skip', 4, 'add_diagonal'),
    ),
)
def test_no_of_etas_linearization(
    tmp_path,
    start_modelres,
    model_count,
    algorithm,
    correlation_algorithm,
    no_of_candidate_models,
    strategy,
):
    with chdir(tmp_path):
        res = run_iivsearch(
            model=start_modelres[0],
            results=start_modelres[1],
            algorithm=algorithm,
            linearize=True,
            correlation_algorithm=correlation_algorithm,
            iiv_strategy=strategy,
        )

        assert res
        # assert len(res.summary_tool) == no_of_candidate_models + 4
        # assert len(res.summary_models) == no_of_candidate_models + 1

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        # assert model_count(rundir) == no_of_candidate_models + 3
        assert (rundir / 'metadata.json').exists()
