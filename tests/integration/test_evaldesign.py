import shutil

from pharmpy import Model
from pharmpy.tools import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_evaldesign(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.create_model('pheno_real.mod')
        model.datainfo = model.datainfo.derive(path=tmp_path / 'pheno.dta')
        res = run_tool('evaldesign', model)
        assert len(res.individual_ofv) == 59
