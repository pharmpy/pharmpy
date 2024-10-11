import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import retrieve_models, run_estmethod
from pharmpy.workflows.contexts import LocalDirectoryContext


@pytest.mark.parametrize(
    'algorithm, methods, parameter_uncertainty_methods, compare_ofv, no_of_candidates, advan_ref',
    [
        ('exhaustive', ['FOCE', 'IMP'], None, True, 2, 'ADVAN2'),
        ('exhaustive_only_eval', ['FOCE', 'IMP'], None, True, 2, 'ADVAN2'),
        ('exhaustive', ['FOCE'], ['SANDWICH', 'SMAT'], False, 2, 'ADVAN2'),
        ('exhaustive_with_update', ['FOCE'], ['SANDWICH', 'SMAT'], False, 4, 'ADVAN2'),
        ('exhaustive_with_update', ['IMP'], ['SANDWICH', 'SMAT'], False, 5, 'ADVAN2'),
    ],
)
def test_estmethod(
    tmp_path,
    start_modelres,
    model_count,
    testdata,
    algorithm,
    methods,
    parameter_uncertainty_methods,
    compare_ofv,
    no_of_candidates,
    advan_ref,
):
    with chdir(tmp_path):
        res = run_estmethod(
            algorithm,
            methods=methods,
            parameter_uncertainty_methods=parameter_uncertainty_methods,
            compare_ofv=compare_ofv,
            model=start_modelres[0],
            results=start_modelres[1],
        )

        assert len(res.summary_tool) == no_of_candidates + 1
        context = LocalDirectoryContext("estmethod1")
        res_models = [model for model in retrieve_models(context) if model.name != 'input_model']
        assert len(res_models) == no_of_candidates
        assert advan_ref in res_models[-1].code
        if compare_ofv:
            assert all(model.execution_steps[-1].evaluation for model in res_models)
        else:
            assert all(model.execution_steps[-1].evaluation is False for model in res_models)
        rundir = tmp_path / 'estmethod1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidates
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
