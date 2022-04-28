import shutil
import tempfile
from pathlib import Path

import pytest

from pharmpy import Model
from pharmpy.modeling import fit
from pharmpy.utils import TemporaryDirectoryChanger


@pytest.fixture(scope='session')
def start_model(testdata):
    tempdir = Path(tempfile.mkdtemp())
    with TemporaryDirectoryChanger(tempdir):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tempdir)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tempdir)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tempdir / 'mox_simulated_normal.csv'
        fit(model_start)
    return model_start
