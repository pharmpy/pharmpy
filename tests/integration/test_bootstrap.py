import shutil

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_bootstrap(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.ext', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.lst', tmp_path)
        model = Model.create_model('pheno.mod')
        model.datainfo.path = tmp_path / 'pheno.dta'
        model.modelfit_results.ofv  # Read in results
        res = run_tool('bootstrap', model, resamples=3)
        assert len(res.parameter_estimates) == 3
