import os
import shutil
from pathlib import Path

import pytest

import pharmpy.modeling as modeling
from pharmpy import Model
from pharmpy.config import site_config_dir, user_config_dir
from pharmpy.plugins.nonmem import conf


class TemporaryDirectoryChanger:
    def __init__(self, path):
        self.path = path
        self.old_path = Path.cwd()

    def __enter__(self):
        os.chdir(self.path)
        return self

    def __exit__(self, *args):
        os.chdir(self.old_path)


def test_configuration():
    print("User config dir:", user_config_dir())
    print("Site config dir:", site_config_dir())
    print("Default NONMEM path:", conf.default_nonmem_path)
    assert (conf.default_nonmem_path / 'license' / 'nonmem.lic').is_file()


def test_fit(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model('pheno.mod')
        model.dataset_path = tmp_path / 'pheno.dta'
        modeling.fit(model)
        rundir = tmp_path / 'modelfit_dir1'
        assert model.modelfit_results.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
