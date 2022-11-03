import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import run_tool


def test_bootstrap(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.ext', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.lst', tmp_path)
        model = Model.create_model('pheno.mod')
        model.datainfo = model.datainfo.derive(path=tmp_path / 'pheno.dta')
        model.modelfit_results.ofv  # Read in results
        res = run_tool('bootstrap', model, resamples=3)
        assert len(res.parameter_estimates) == 3
