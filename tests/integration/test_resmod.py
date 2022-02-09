import shutil

from pharmpy import Model
from pharmpy.modeling import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_resmod(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.create_model('pheno_real.mod')
        model.datainfo.path = tmp_path / 'pheno.dta'
        res = run_tool('resmod', model, groups=4, p_value=0.05, skip=[])
        assert res
