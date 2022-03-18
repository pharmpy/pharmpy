import shutil
from pathlib import Path

import pharmpy.modeling
from pharmpy import Model
from pharmpy.modeling import run_amd, run_iiv
from pharmpy.utils import TemporaryDirectoryChanger


def test_amd(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        dipath = Path(pharmpy.modeling.__file__).parent / 'example_models' / 'pheno.datainfo'
        shutil.copy2(dipath, tmp_path)

        res = run_amd(tmp_path / 'pheno.dta', mfl='LAGTIME();PERIPHERALS(1)')
        assert res


def test_iiv(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model.create_model('mox2.mod')
        model_start.datainfo.path = tmp_path / 'mx19B.csv'

        res = run_iiv(model_start)

        assert len(res.summary_tool) == 2
        assert len(res.summary_tool[0]) == 7
        assert len(res.summary_tool[1]) == 4
        assert len(res.summary_models) == 12
        assert len(res.models) == 11
        rundir1 = tmp_path / 'iiv_dir1'
        assert rundir1.is_dir()
        assert len(list((rundir1 / 'models').iterdir())) == 8
        rundir2 = tmp_path / 'iiv_dir2'
        assert rundir2.is_dir()
        assert len(list((rundir2 / 'models').iterdir())) == 4
