import shutil

from pharmpy import Model
from pharmpy.tools.resmod import Resmod
from pharmpy.utils import TemporaryDirectoryChanger


def test_resmod(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model('pheno_real.mod')
        model.dataset_path = tmp_path / 'pheno.dta'
        model.modelfit_results.residuals  # FIXME: Shouldn't be needed
        res = Resmod(model).run()
        assert res
