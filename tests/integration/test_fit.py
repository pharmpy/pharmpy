import os
import shutil

import pharmpy.modeling as modeling
from pharmpy import Model
from pharmpy.config import site_config_dir, user_config_dir
from pharmpy.plugins.nonmem import conf


def test_configuration():
    print("User config dir:", user_config_dir())
    print("Site config dir:", site_config_dir())
    print(conf.default_nonmem_path)
    assert (conf.default_nonmem_path / 'license' / 'nonmem.lic').is_file()


def test_fit(tmp_path, testdata):
    os.chdir(tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
    model = Model('pheno.mod')
    modeling.fit(model)
    rundir = tmp_path / 'modelfit_dir1'
    assert rundir.is_dir()
