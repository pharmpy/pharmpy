import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import open_project, read_modelfit_results, run_tool


def test_bootstrap(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.ext', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.lst', tmp_path)
        model = Model.parse_model('pheno.mod')
        results = read_modelfit_results('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = run_tool('bootstrap', model=model, results=results, samples=3, seed=12345)
        assert len(res.parameter_estimates) == 3


def test_bootstrap_project(tmp_path, testdata, model_count):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.ext', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.lst', tmp_path)
        model = Model.parse_model('pheno.mod')
        results = read_modelfit_results('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))

        proj = open_project('myproject', tmp_path)

        res1 = run_tool(
            'bootstrap', model=model, results=results, samples=3, seed=12345, project=proj
        )
        assert model_count(proj.model_database.path, '') == 4
        res2 = run_tool(
            'bootstrap', model=model, results=results, samples=5, seed=12345, project=proj
        )
        assert model_count(proj.model_database.path, '') == 6
        assert res1.included_individuals == res2.included_individuals[:3]
