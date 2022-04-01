import shutil

import pytest

import pharmpy.modeling as modeling
from pharmpy import Model
from pharmpy.config import site_config_dir, user_config_dir
from pharmpy.plugins.nonmem import conf
from pharmpy.utils import TemporaryDirectoryChanger


def test_configuration():
    print("User config dir:", user_config_dir())
    print("Site config dir:", site_config_dir())
    print("Default NONMEM path:", conf.default_nonmem_path)
    assert (conf.default_nonmem_path / 'license' / 'nonmem.lic').is_file()


def test_fit_single(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.create_model('pheno.mod')
        model.datainfo.path = tmp_path / 'pheno.dta'
        modeling.fit(model)
        rundir = tmp_path / 'modelfit_dir1'
        assert model.modelfit_results.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 2


def test_fit_multiple(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno_1.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno_1.dta')
        model_1 = Model.create_model('pheno_1.mod')
        model_1.datainfo.path = tmp_path / 'pheno_1.dta'
        model_1.update_source()
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno_2.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno_2.dta')
        model_2 = Model.create_model('pheno_2.mod')
        model_2.datainfo.path = tmp_path / 'pheno_2.dta'
        model_2.update_source()
        modeling.fit([model_1, model_2])
        rundir = tmp_path / 'modelfit_dir1'
        assert model_1.modelfit_results.ofv == pytest.approx(730.8947268137308)
        assert model_2.modelfit_results.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
        assert len(list((rundir / 'models').iterdir())) == 3


def test_fit_copy(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno.dta')

        model_1 = Model.create_model('pheno.mod')
        model_1.datainfo.path = tmp_path / 'pheno.dta'
        modeling.fit(model_1)

        rundir_1 = tmp_path / 'modelfit_dir1'
        assert rundir_1.is_dir()
        assert len(list((rundir_1 / 'models').iterdir())) == 2

        model_2 = modeling.copy_model(model_1, 'pheno_copy')
        modeling.update_inits(model_2)
        modeling.fit(model_2)

        rundir_2 = tmp_path / 'modelfit_dir1'
        assert rundir_2.is_dir()
        assert len(list((rundir_2 / 'models').iterdir())) == 2

        assert model_1.modelfit_results.ofv != model_2.modelfit_results.ofv
        mod_files = [
            path for path in tmp_path.iterdir() if path.is_file() and path.suffix == '.mod'
        ]
        assert len(mod_files) == 2


def test_fit_nlmixr(tmp_path, testdata):
    from pharmpy.plugins.nlmixr import conf

    if str(conf.rpath) == '.':
        pytest.skip("No R selected in conf. Skipping nlmixr tests")
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.create_model('pheno.mod')
        model.datainfo.path = tmp_path / 'pheno.dta'
        model = modeling.convert_model(model, 'nlmixr')
        modeling.fit(model, tool='nlmixr')
        assert model.modelfit_results.ofv == pytest.approx(732.58737)
        assert model.modelfit_results.parameter_estimates['TVCL'] == pytest.approx(0.0058606648)
