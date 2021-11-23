import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'list_of_parameters, no_of_models',
    [([], 1), (['S1'], 4)],
)
def test_iiv(tmp_path, testdata, list_of_parameters, no_of_models):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model('pheno_real.mod')
        model.dataset_path = tmp_path / 'pheno.dta'

        add_iiv(model, list_of_parameters, 'add')

        res = run_tool('iiv', 'brute_force', model=model)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in range(570, 590) for model in res.models)
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1
