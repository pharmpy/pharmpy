from functools import partial

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_iov, read_model
from pharmpy.tools import fit, run_qa


@pytest.mark.parametrize(
    'funcs, kwargs',
    [
        ([], {'linearize': False}),
        ([], {'linearize': True}),
        ([partial(add_iov, occ='VISI', list_of_parameters=['CL'])], {'linearize': False}),
    ],
)
def test_qa(testdata, tmp_path, model_count, start_modelres, funcs, kwargs):
    with chdir(tmp_path):
        path = testdata / 'nonmem' / 'models' / 'mox2.mod'
        model = read_model(path)
        for func in funcs:
            model = func(model)
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
