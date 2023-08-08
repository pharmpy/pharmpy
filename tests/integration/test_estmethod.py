import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.tools import run_estmethod


@pytest.mark.parametrize(
    'algorithm, methods, covs, no_of_candidates, advan_ref',
    [
        ('exhaustive', ['foce', 'imp'], None, 2, 'ADVAN2'),
        ('exhaustive_only_eval', ['foce', 'imp'], None, 2, 'ADVAN2'),
        ('exhaustive', ['foce'], ['sandwich', 'cpg'], 2, 'ADVAN2'),
    ],
)
def test_estmethod(
    tmp_path,
    start_model,
    model_count,
    testdata,
    algorithm,
    methods,
    covs,
    no_of_candidates,
    advan_ref,
):
    with chdir(tmp_path):
        res = run_estmethod(algorithm, methods=methods, covs=covs, model=start_model)

        assert len(res.summary_tool) == no_of_candidates
        assert len(res.models) == no_of_candidates
        assert advan_ref in res.models[-1].model_code
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidates
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
