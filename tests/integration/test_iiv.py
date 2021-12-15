import shutil

import pytest

from pharmpy import Model
from pharmpy.modeling import add_iiv, add_peripheral_compartment, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.parametrize(
    'list_of_parameters, no_of_models',
    [([], 4), (['QP1'], 14)],
)
def test_iiv_block_structure(tmp_path, testdata, list_of_parameters, no_of_models):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'

        add_peripheral_compartment(model_start)
        add_iiv(model_start, list_of_parameters, 'add')

        res = run_tool('iiv', 'brute_force_block_structure', model=model_start)

        assert len(res.summary) == no_of_models
        assert len(res.models) == no_of_models
        assert all(int(model.modelfit_results.ofv) in range(-2200, -2000) for model in res.models)
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == no_of_models + 1


def test_iiv_no_of_etas(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'

        res = run_tool('iiv', 'brute_force_no_of_etas', model=model_start)

        assert len(res.summary) == 7
        assert len(res.models) == 7
        assert all(int(model.modelfit_results.ofv) in range(-1900, -1400) for model in res.models)
        rundir = tmp_path / 'iiv_dir1'
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 8
