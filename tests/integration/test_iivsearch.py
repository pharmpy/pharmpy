import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import (
    add_lag_time,
    convert_model,
    create_basic_pk_model,
    set_direct_effect,
    set_estimation_step,
)
from pharmpy.tools import fit, run_iivsearch
from pharmpy.workflows import LocalDirectoryContext


@pytest.mark.parametrize(
    'algorithm, correlation_algorithm, mfl, kwargs, no_of_models, no_of_steps, max_diff_params, best_model',
    [
        (
            'top_down_exhaustive',
            'skip',
            'IIV?(@PK,exp)',
            dict(),
            8,
            2,
            3,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            'skip',
            'IIV(CL,exp);IIV?(@PK,exp)',
            dict(),
            4,
            2,
            2,
            'iivsearch_run1',
        ),
        (
            'skip',
            'top_down_exhaustive',
            'IIV(@PK,exp);COVARIANCE?(IIV,@IIV)',
            dict(),
            5,
            2,
            3,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            'skip',
            'IIV(CL,exp);IIV?(@PK,[exp,add])',
            dict(),
            9,
            2,
            2,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            None,
            'IIV(CL,exp);IIV?(@PK,exp);COVARIANCE?(IIV,@IIV)',
            dict(),
            5,
            3,
            2,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            'skip',
            'IIV?(@PK,exp)',
            {'as_fullblock': True},
            17,
            2,
            10,
            'iivsearch_run15',
        ),
        (
            'top_down_exhaustive',
            None,
            'IIV(CL,exp);IIV?(@PK,[exp,add]);COVARIANCE?(IIV,@IIV)',
            {'as_fullblock': True},
            29,
            3,
            9,
            'iivsearch_run27',
        ),
        (
            'bottom_up_stepwise',
            'top_down_exhaustive',
            'IIV(CL,exp);IIV?(@PK,exp);COVARIANCE?(IIV,@IIV)',
            dict(),
            6,
            4,
            1,
            'iivsearch_run1',
        ),
        (
            'bottom_up_stepwise',
            'top_down_exhaustive',
            'IIV(CL,exp);IIV?(@PK,exp);COVARIANCE?(IIV,@IIV)',
            {'as_fullblock': True},
            19,
            5,
            6,
            'iivsearch_run7',
        ),
        (
            'top_down_exhaustive',
            'skip',
            'IIV(CL,exp);IIV?(@PK,exp)',
            {
                'parameter_uncertainty_method': 'SANDWICH',
                'strictness': 'minimization_successful and rse <= 0.5',
            },
            4,
            2,
            2,
            'iivsearch_run1',
        ),
        (
            'bottom_up_stepwise',
            'skip',
            'IIV(CL,exp);IIV?(@PK,exp)',
            {
                'parameter_uncertainty_method': 'SANDWICH',
                'strictness': 'minimization_successful and rse <= 0.5',
            },
            5,
            3,
            1,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            None,
            'IIV(CL,exp);IIV?(@PK,exp)',
            {'rank_type': 'mbic', 'E_p': '50%', 'E_q': '50%'},
            4,
            2,
            2,
            'iivsearch_run1',
        ),
        (
            'bottom_up_stepwise',
            None,
            'IIV(CL,exp);IIV?(@PK,exp)',
            {'rank_type': 'mbic', 'E_p': '50%', 'E_q': '50%'},
            5,
            3,
            1,
            'iivsearch_run1',
        ),
    ],
)
def test_iivsearch_dummy(
    tmp_path,
    model_count,
    start_modelres_dummy,
    algorithm,
    correlation_algorithm,
    mfl,
    kwargs,
    no_of_models,
    no_of_steps,
    max_diff_params,
    best_model,
):
    with chdir(tmp_path):
        has_iiv_strategy = 'as_fullblock' in kwargs and kwargs['as_fullblock']
        if has_iiv_strategy:
            start_model = add_lag_time(start_modelres_dummy[0])
            start_res = fit(start_model, esttool='dummy')
        else:
            start_model = start_modelres_dummy[0]
            start_res = start_modelres_dummy[1]

        res = run_iivsearch(
            model=start_model,
            results=start_res,
            algorithm=algorithm,
            correlation_algorithm=correlation_algorithm,
            search_space=mfl,
            esttool='dummy',
            **kwargs,
        )

        assert len(res.summary_tool.index.get_level_values('model').unique()) == no_of_models
        assert len(res.summary_models) == no_of_models
        assert len(res.summary_tool.index.get_level_values('step').unique()) == no_of_steps
        assert res.summary_tool['d_params'].abs().max() == max_diff_params

        assert res.final_model.name == best_model

        ctx = LocalDirectoryContext('iivsearch1')
        cand_model = ctx.retrieve_model_entry('iivsearch_run2').model
        assert cand_model.random_variables != start_model.random_variables

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_models + 1  # Include final
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
        assert (rundir / 'models' / 'iivsearch_run1' / 'model_results.json').exists()
        assert not (rundir / 'models' / 'iivsearch_run1' / 'model.lst').exists()


@pytest.mark.parametrize(
    'algorithm, correlation_algorithm, mfl, no_of_models, max_diff_params, best_model',
    [
        (
            'top_down_exhaustive',
            'skip',
            'IIV(@PK,exp);IIV?(@PD,add)',
            4,
            2,
            'iivsearch_run1',
        ),
        (
            'skip',
            'top_down_exhaustive',
            'IIV(@PK,exp);IIV(@PD,exp);COVARIANCE(IIV,[CL,VC]);COVARIANCE?(IIV,@PD_IIV)',
            3,
            3,
            'iivsearch_run1',
        ),
        (
            'top_down_exhaustive',
            None,
            'IIV(@PK,exp);IIV?(@PD,add);COVARIANCE(IIV,@PK_IIV);COVARIANCE?(IIV,@PD_IIV)',
            5,
            3,
            'iivsearch_run4',
        ),
    ],
)
def test_iivsearch_pd_dummy(
    tmp_path,
    load_model_for_test,
    testdata,
    model_count,
    start_modelres_dummy,
    algorithm,
    correlation_algorithm,
    mfl,
    no_of_models,
    max_diff_params,
    best_model,
):
    with chdir(tmp_path):
        pk_model = create_basic_pk_model('iv', testdata / 'nonmem' / 'pheno_pd.csv')
        pk_model = convert_model(pk_model, to_format='nonmem')
        pd_model = set_direct_effect(pk_model, 'linear')
        pd_res = fit(pd_model, esttool='dummy')

        res = run_iivsearch(
            model=pd_model,
            results=pd_res,
            algorithm=algorithm,
            correlation_algorithm=correlation_algorithm,
            search_space=mfl,
            esttool='dummy',
        )

        assert len(res.summary_tool.index.get_level_values('model').unique()) == no_of_models
        assert len(res.summary_models) == no_of_models
        no_of_steps = 2 if 'skip' in (algorithm, correlation_algorithm) else 3
        assert len(res.summary_tool.index.get_level_values('step').unique()) == no_of_steps
        assert res.summary_tool['d_params'].abs().max() == max_diff_params

        assert res.final_model.name == best_model


@pytest.mark.filterwarnings(
    'ignore::UserWarning',
)
@pytest.mark.parametrize(
    ('algorithm', 'correlation_algorithm', 'no_of_linearized_models', 'kwargs'),
    (
        (
            'top_down_exhaustive',
            'skip',
            4,
            {'search_space': 'IIV(CL,exp);IIV?(@PK,exp)'},
        ),
        (
            'top_down_exhaustive',
            None,
            10,
            {'search_space': 'IIV(CL,exp);IIV?(@PK,exp);COVARIANCE?(IIV,@IIV)'},
        ),
        (
            'skip',
            'top_down_exhaustive',
            5,
            {'search_space': 'IIV(@PK,exp);COVARIANCE?(IIV,@IIV)'},
        ),
        (
            'bottom_up_stepwise',
            'skip',
            4,
            {'search_space': 'IIV(CL,exp);IIV?(@PK,exp)'},
        ),
    ),
)
def test_iivsearch_linearization(
    tmp_path,
    load_model_for_test,
    testdata,
    model_count,
    algorithm,
    correlation_algorithm,
    no_of_linearized_models,
    kwargs,
):
    with chdir(tmp_path):
        model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
        model = set_estimation_step(model, method='FOCE', interaction=True)
        mfr = fit(model)
        res = run_iivsearch(
            model=model,
            results=mfr,
            algorithm=algorithm,
            correlation_algorithm=correlation_algorithm,
            linearize=True,
            **kwargs,
        )

        assert res

        no_of_models = len(res.summary_tool.index.get_level_values('model').unique())
        assert no_of_models == no_of_linearized_models + 2

        rundir = tmp_path / 'iivsearch1'
        assert rundir.is_dir()
        # assert model_count(rundir) == no_of_candidate_models + 3
        assert (rundir / 'metadata.json').exists()
