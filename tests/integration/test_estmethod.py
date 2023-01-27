import shutil

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import set_zero_order_elimination
from pharmpy.tools import run_estmethod


@pytest.mark.parametrize(
    'algorithm, methods, solvers, no_of_candidates, advan_ref',
    [
        ('exhaustive', ['foce', 'imp'], None, 2, 'ADVAN1'),
        ('exhaustive_only_eval', ['foce', 'imp'], None, 2, 'ADVAN1'),
    ],
)
def test_estmethod(
    tmp_path, model_count, testdata, algorithm, methods, solvers, no_of_candidates, advan_ref
):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model_start = Model.create_model('pheno_real.mod')
        model_start.datainfo = model_start.datainfo.replace(path=tmp_path / 'pheno.dta')
        set_zero_order_elimination(model_start)

        res = run_estmethod(algorithm, methods=methods, solvers=solvers, model=model_start)

        assert len(res.summary_tool) == no_of_candidates
        assert len(res.models) == no_of_candidates
        assert advan_ref in res.models[-1].model_code
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidates
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
