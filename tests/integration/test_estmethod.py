import shutil
from pathlib import Path

import pytest

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


@pytest.mark.parametrize(
    'algorithm, methods, solvers, no_of_models, advan_ref',
    [
        ('exhaustive', 'imp', None, 3, 'ADVAN1'),
        ('exhaustive', 'imp', ['lsoda'], 5, 'ADVAN1'),
        ('reduced', ['foce', 'imp'], None, 2, 'ADVAN1'),
    ],
)
def test_estmethod(tmp_path, testdata, algorithm, methods, solvers, no_of_models, advan_ref):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model_start = Model.create_model('pheno_real.mod')
        model_start.datainfo.path = tmp_path / 'pheno.dta'

        res = run_tool('estmethod', algorithm, methods=methods, solvers=solvers, model=model_start)

        assert len(res.summary_tool) == no_of_models
        assert len(res.models) == no_of_models
        assert advan_ref in res.models[-1].model_code
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert _model_count(rundir) == no_of_models
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
