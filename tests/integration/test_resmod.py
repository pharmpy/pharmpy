import shutil

from pharmpy import Model
from pharmpy.modeling import remove_covariance_step, run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_resmod(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem' / 'resmod').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'resmod' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'resmod' / 'mytab', tmp_path)

        model = Model.create_model('mox3.mod')
        remove_covariance_step(model)
        model.datainfo.path = tmp_path / 'moxo_simulated_resmod.csv'
        res = run_tool('resmod', model, groups=4, p_value=0.05, skip=[])
        iteration = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        assert (res.models.index.get_level_values('iteration') == iteration).all()
        assert res.best_model.model_code.split('\n')[11] == 'IF (TAD.LT.6.08) THEN'
        assert (
            res.best_model.model_code.split('\n')[12]
            == '    Y = A(2)*EPS(1)*THETA(4)*EXP(ETA(4))/VC + A(2)/VC'
        )
        assert (
            res.best_model.model_code.split('\n')[14]
            == '    Y = A(2)*EPS(1)*EXP(ETA(4))/VC + A(2)/VC'
        )
        assert res.best_model.model_code.split('\n')[19] == '$THETA  1.15573 ; time_varying'
        assert res.best_model.model_code.split('\n')[25] == '$OMEGA  0.0396751 ; IIV_RUV1'
