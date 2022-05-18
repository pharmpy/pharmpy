import shutil

from pharmpy import Model
from pharmpy.modeling import remove_covariance_step, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_resmod(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.create_model('pheno_real.mod')
        remove_covariance_step(model)
        model.datainfo.path = tmp_path / 'pheno.dta'
        del model.statements[0:2]
        res = run_tool('resmod', model, groups=4, p_value=0.5, skip=[])
        assert res.best_model.model_code.split('\n')[15] == 'IF (TAD.LT.11.5) THEN'
        assert res.best_model.model_code.split('\n')[16] == '    Y = EPS(1)*THETA(4)*W + F'
        assert res.best_model.model_code.split('\n')[27] == '$THETA  0.807384 ; time_varying'
