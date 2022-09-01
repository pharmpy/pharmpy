import shutil
from pathlib import Path

import pytest

from pharmpy.tools import run_amd
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_amd(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.datainfo', tmp_path)
        input = (tmp_path / 'mox_simulated_normal.csv').as_posix()
        run_amd(
            input,
            modeltype='pk_oral',
            search_space='PERIPHERALS(1)',
            occasion='VISI',
        )
        rundir = tmp_path / 'amd_dir1'
        assert rundir.is_dir()
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        subrundir = [
            'modelfit',
            'modelsearch',
            'iivsearch',
            'ruvsearch',
            'iovsearch',
            'allometry',
            'covsearch',
        ]
        for dir in subrundir:
            dir = rundir / dir
            assert _model_count(dir) >= 1


def _model_count(rundir: Path):
    return sum(
        map(
            lambda path: 0 if path.name in ['.lock', '.datasets'] else 1,
            ((rundir / 'models').iterdir()),
        )
    )
