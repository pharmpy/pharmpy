import shutil

import pytest

from pharmpy.tools import run_amd
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_amd(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.datainfo', tmp_path)
        input = (tmp_path / 'mox_simulated_normal.csv').as_posix()
        res = run_amd(
            input,
            modeltype='pk_oral',
            search_space='PERIPHERALS(1)',
            occasion='VISI',
        )
        assert len(res.summary_tool) == 58
        assert len(res.summary_models) == 60
        assert len(res.summary_individuals_count) == 60
        assert res.final_model.name == 'scaled_model'
