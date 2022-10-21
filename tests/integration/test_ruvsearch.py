import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import remove_covariance_step
from pharmpy.tools import retrieve_final_model, run_tool


def test_ruvsearch(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        model = Model.create_model('mox3.mod')
        remove_covariance_step(model)
        model.datainfo = model.datainfo.derive(path=tmp_path / 'moxo_simulated_resmod.csv')
        res = run_tool(
            'ruvsearch', model, results=model.modelfit_results, groups=4, p_value=0.05, skip=[]
        )
        iteration = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        best_model = retrieve_final_model(res)
        assert (res.cwres_models.index.get_level_values('iteration') == iteration).all()
        assert best_model.model_code.split('\n')[11] == 'IF (TAD.LT.6.08) THEN'
        assert (
            best_model.model_code.split('\n')[12]
            == '    Y = A(2)*EPS(1)*THETA(4)*EXP(ETA(4))/VC + A(2)/VC'
        )
        assert (
            best_model.model_code.split('\n')[14] == '    Y = A(2)*EPS(1)*EXP(ETA(4))/VC + A(2)/VC'
        )
        assert best_model.model_code.split('\n')[19] == '$THETA  1.15573 ; time_varying'
        assert best_model.model_code.split('\n')[25] == '$OMEGA  0.0396751 ; IIV_RUV1'
