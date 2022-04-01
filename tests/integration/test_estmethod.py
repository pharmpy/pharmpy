import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'methods, solvers, no_of_models, advan_ref',
    [('foce', None, 2, 'ADVAN1'), ('foce', ['lsoda'], 4, 'ADVAN13'), ('foce', ['GL'], 4, '$MODEL')],
)
def test_estmethod(tmp_path, testdata, methods, solvers, no_of_models, advan_ref):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model_start = Model.create_model('pheno_real.mod')
        model_start.datainfo.path = tmp_path / 'pheno.dta'

        res = run_tool('estmethod', methods=methods, solvers=solvers, model=model_start)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert advan_ref in res.models[-1].model_code
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'results.html').exists()
