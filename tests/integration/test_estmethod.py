import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'method',
    ['fo', 'laplace'],
)
def test_estmethod(tmp_path, testdata, method):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model_start = Model('pheno_real.mod')
        model_start.dataset_path = tmp_path / 'pheno.dta'

        res = run_tool('estmethod', methods=method, model=model_start)

        no_of_models = 4
        assert len(res.summary) == no_of_models * 2
        assert len(res.models) == no_of_models
        rundir = tmp_path / 'estmethod_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models
