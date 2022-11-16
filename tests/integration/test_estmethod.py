import shutil

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import run_estmethod


@pytest.mark.filterwarnings('ignore:.*Expected result files do not exist')
@pytest.mark.parametrize(
    'algorithm, methods, solvers, no_of_candidates, advan_ref',
    [
        ('exhaustive', ['imp'], None, 3, 'ADVAN1'),
        ('exhaustive', ['imp'], ['lsoda'], 3, 'ADVAN13'),
        ('reduced', ['foce', 'imp'], None, 2, 'ADVAN1'),
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
        model_start.datainfo = model_start.datainfo.derive(path=tmp_path / 'pheno.dta')

        res = run_estmethod(algorithm, methods=methods, solvers=solvers, model=model_start)

        if algorithm == 'reduced':
            no_of_models = no_of_candidates + 1
        else:
            no_of_models = no_of_candidates

        assert len(res.summary_tool) == no_of_models
        assert len(res.models) == no_of_candidates
        assert advan_ref in res.models[-1].model_code
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert model_count(rundir) == no_of_candidates
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
