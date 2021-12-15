import shutil

from pharmpy import Model
from pharmpy.modeling import run_iiv, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_amd(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model('pheno_real.mod')
        model.dataset_path = tmp_path / 'pheno.dta'

        # FIXME: remove after updating results
        model.modelfit_results.estimation_step

        res = run_tool('amd', model)
        assert res


def test_iiv(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mx19B.csv', tmp_path)
        model_start = Model('mox2.mod')
        model_start.dataset_path = tmp_path / 'mx19B.csv'

        res = run_iiv(model_start)

        assert len(res.summary) == 11
        assert len(res.models) == 11
        rundir1 = tmp_path / 'iiv_dir1'
        assert rundir1.is_dir()
        assert len(list((rundir1 / 'models').iterdir())) == 8
        rundir2 = tmp_path / 'iiv_dir2'
        assert rundir2.is_dir()
        assert len(list((rundir2 / 'models').iterdir())) == 4
