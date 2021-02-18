import os
import shutil

from pharmpy import Model
import pharmpy.modeling as modeling


def test_fit(tmp_path, testdata):
    os.chdir(tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
    model = Model('pheno.mod')
    modeling.fit(model)
    rundir = tmp_path / 'modelfit_dir1'
    assert rundir.is_dir()
