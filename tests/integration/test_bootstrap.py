import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, run_tool


def test_bootstrap(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.ext', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.lst', tmp_path)
        model = Model.parse_model('pheno.mod')
        results = read_modelfit_results('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = run_tool('bootstrap', model=model, results=results, resamples=3)
        assert len(res.parameter_estimates) == 3
