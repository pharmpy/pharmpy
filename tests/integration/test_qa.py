import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import read_model
from pharmpy.tools import fit, run_qa


@pytest.mark.parametrize(
    'kwargs',
    [
        {'linearize': False},
        {'linearize': True},
    ],
)
def test_qa(testdata, tmp_path, model_count, start_modelres, kwargs):
    with chdir(tmp_path):
        path = testdata / 'nonmem' / 'pheno_real.mod'
        model = read_model(path)
        mfr = fit(model)
        res = run_qa(model=model, results=mfr, **kwargs)

        assert res.tdist_plot
        assert res.boxcox_plot

        rundir = tmp_path / 'qa1'
        assert model_count(rundir) == 4

        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
        assert (rundir / 'metadata.json').exists()
