import shutil

from pharmpy import Model
from pharmpy.modeling import run_tool
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
