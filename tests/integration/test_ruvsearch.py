import shutil

import pytest

from pharmpy.model import Model
from pharmpy.modeling import remove_covariance_step
from pharmpy.tools import run_tool
from pharmpy.utils import TemporaryDirectoryChanger


def test_ruvsearch(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        model = Model.create_model('mox3.mod')
        remove_covariance_step(model)
        model.datainfo = model.datainfo.derive(path=tmp_path / 'moxo_simulated_resmod.csv')
        res = run_tool('ruvsearch', model, groups=4, p_value=0.05, skip=[])
        iteration = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        assert (res.cwres_models.index.get_level_values('iteration') == iteration).all()
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


def test_ruvsearch_input(tmp_path, testdata):
    with TemporaryDirectoryChanger(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        model = Model.create_model('mox3.mod')
        remove_covariance_step(model)
        model.datainfo = model.datainfo.derive(path=tmp_path / 'moxo_simulated_resmod.csv')

        res = run_tool('ruvsearch', model)
        assert res

        with pytest.raises(TypeError) as e:
            run_tool('ruvsearch', model, groups=4.5, p_value=0.05, skip=[])
        assert (
            str(e.value)
            == '4.5 is not an integer. Please input an integer for groups and try again.'
        )

        with pytest.raises(ValueError) as e:
            run_tool('ruvsearch', model, groups=4, p_value=1.2, skip=[])
        assert (
            str(e.value)
            == '1.2 is not a float number between (0, 1). Please input correct p-value and try again.'
        )

        with pytest.raises(ValueError) as e:
            run_tool(
                'ruvsearch',
                model,
                groups=4,
                p_value=0.05,
                skip=['tume_varying', 'RUV_IIV', 'powder'],
            )
        assert str(e.value) == "Please correct ['tume_varying', 'RUV_IIV', 'powder'] and try again."

        with pytest.raises(ValueError) as e:
            del model.modelfit_results.residuals['CWRES']
            run_tool('ruvsearch', model, groups=4, p_value=0.05, skip=[])
        assert (
            str(e.value) == 'Please check mox3.mod file to make sure ID, TIME, CWRES are in $TABLE.'
        )
