import shutil
from pathlib import Path

from pharmpy import Model
from pharmpy.modeling import run_iovsearch
from pharmpy.utils import TemporaryDirectoryChanger


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )


def test_default_mox2(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iovsearch('VISI', model=start_model)
        rundir = tmp_path / 'iovsearch_dir1'
        assert _model_count(rundir) == 8

        assert res.best_model.name == 'mox2-with-matching-IOVs-remove_iov-6'


def test_rank_type_ofv_mox2(tmp_path, start_model):
    with TemporaryDirectoryChanger(tmp_path):
        res = run_iovsearch('VISI', model=start_model, rank_type='ofv')
        rundir = tmp_path / 'iovsearch_dir1'
        assert _model_count(rundir) == 8

        assert res.best_model.name == 'mox2-with-matching-IOVs-remove_iov-6'


def test_default_mox1(tmp_path, testdata):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox1.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_log.csv', tmp_path)
    with TemporaryDirectoryChanger(tmp_path):
        start_model = Model.create_model('mox1.mod')
        res = run_iovsearch('VISI', model=start_model)
        rundir = tmp_path / 'iovsearch_dir1'
        assert _model_count(rundir) == 8

        assert res.best_model == res.input_model
