import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import run_tool


def test_evaldesign(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.parse_model('pheno_real.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = run_tool('evaldesign', model)
        assert len(res.individual_ofv) == 59
