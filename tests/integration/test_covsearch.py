import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_covsearch, run_tool


@pytest.mark.parametrize(
    'search_space, algorithm, no_of_effects, best_model',
    [
        (
            'COVARIATE?([CL,MAT,VC],[AGE,WT],exp,*)',
            'scm-forward',
            6,
            'covsearch_run21',
        ),
        (
            'COVARIATE?([CL,VC],[AGE,WT],exp,*)',
            'scm-forward',
            4,
            'covsearch_run10',
        ),
        (
            'LET(CONTINUOUS,[AGE,WT])\n' 'COVARIATE?([CL,VC],@CONTINUOUS,exp,*)',
            'scm-forward',
            4,
            'covsearch_run10',
        ),
        (
            'COVARIATE?([CL,VC],[AGE,WT],exp,*)',
            'scm-forward-then-backward',
            4,
            'covsearch_run18',
        ),
    ],
)
def test_covsearch_dummy(
    tmp_path,
    model_count,
    start_modelres_dummy,
    search_space,
    algorithm,
    no_of_effects,
    best_model,
):
    with chdir(tmp_path):
        res = run_covsearch(
            model=start_modelres_dummy[0],
            results=start_modelres_dummy[1],
            search_space=search_space,
            algorithm=algorithm,
            esttool='dummy',
        )
        no_of_models = (no_of_effects * (no_of_effects + 1)) / 2
        if algorithm == 'scm-forward-then-backward':
            no_of_models += (2 * no_of_effects) + 1

        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1
        assert res.summary_tool['d_params'].max() == no_of_effects

        step_1_description = res.summary_tool.loc[1]['description'].tolist()
        step_1_sorted = sorted(step_1_description)
        assert step_1_description == step_1_sorted

        assert res.final_model.name == best_model

        rundir = tmp_path / 'covsearch1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_models + 2
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
        assert (rundir / 'models' / 'covsearch_run1' / 'model_results.json').exists()
        assert not (rundir / 'models' / 'covsearch_run1' / 'model.lst').exists()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_default_str(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        search_space = (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)'
        )
        run_tool(
            'covsearch',
            model=start_modelres[0],
            results=start_modelres[1],
            search_space=search_space,
        )

        rundir = tmp_path / 'covsearch1'
        assert model_count(rundir) == 39 + 2


def test_covsearch_dummy_adaptive_scope_reduction(tmp_path, start_modelres):
    with chdir(tmp_path):
        search_space = (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)'
        )
        p_value = 0.0001
        res = run_covsearch(
            model=start_modelres[0],
            results=start_modelres[1],
            search_space=search_space,
            algorithm='scm-forward',
            esttool='dummy',
            adaptive_scope_reduction=True,
            p_forward=p_value,
        )

        df_step_1 = res.summary_tool.loc[1]
        step_1_desc_nonsig = df_step_1.loc[df_step_1['pvalue'] > p_value]['description']
        df_step_2 = res.summary_tool.loc[2]
        step_2_desc_all = ';'.join(df_step_2['description'])
        for desc in step_1_desc_nonsig:
            assert desc not in step_2_desc_all

        no_of_models = 36
        assert len(res.summary_tool) == no_of_models + 1
        assert len(res.summary_models) == no_of_models + 1

        assert res.final_model.name == 'covsearch_run36'


def test_adaptive_scope_reduction(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        search_space = (
            'LET(CONTINUOUS, [AGE, WT]); LET(CATEGORICAL, SEX)\n'
            'COVARIATE?([CL, MAT, VC], @CONTINUOUS, exp, *)\n'
            'COVARIATE?([CL, MAT, VC], @CATEGORICAL, cat, *)'
        )
        run_tool(
            'covsearch',
            model=start_modelres[0],
            results=start_modelres[1],
            search_space=search_space,
            adaptive_scope_reduction=True,
        )

        rundir = tmp_path / 'covsearch1'
        assert model_count(rundir) == 33 + 2
