import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import (  # convert_model,; create_basic_pk_model,
    remove_parameter_uncertainty_step,
    transform_blq,
)
from pharmpy.tools import read_modelfit_results, run_tool  # fit,; run_structsearch,
from pharmpy.workflows import LocalDirectoryContext

# from pharmpy.tools.structsearch.pkpd import create_pk_model


# def test_ruvsearch_dv(tmp_path, load_model_for_test, testdata):
#    with chdir(tmp_path):
#        dataset = testdata / "nonmem/pheno_pd.csv"
#        model = create_basic_pk_model("iv", dataset_path=dataset)
#        model = convert_model(model, 'nonmem')

# Fit pk model
#        pk_model = create_pk_model(model)
#        pk_res = fit(pk_model)

# structsearch
#        res_struct = run_structsearch(type='pkpd', results=pk_res, model=model)
#        model = res_struct.models[1]

# ruvsearch
#        res_ruv = run_tool('ruvsearch', model=model, results=model.modelfit_results, dv=2)
#        assert res_ruv


def test_ruvsearch(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        model = Model.parse_model('mox3.mod')
        results = read_modelfit_results('mox3.mod')
        model = remove_parameter_uncertainty_step(model)
        res = run_tool('ruvsearch', model=model, results=results, groups=4, p_value=0.05, skip=[])
        iteration = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
        ctx = LocalDirectoryContext("ruvsearch1")
        best_model = ctx.retrieve_final_model_entry().model
        assert (res.cwres_models.index.get_level_values('iteration') == iteration).all()
        assert best_model.code.split('\n')[12] == 'IF (TAD.LT.6.08) THEN'
        assert (
            best_model.code.split('\n')[13]
            == '    Y = A(2)/VC + A(2)*EPS(1)*THETA(4)*EXP(ETA_RV1)/VC'
        )
        assert best_model.code.split('\n')[15] == '    Y = A(2)/VC + A(2)*EPS(1)*EXP(ETA_RV1)/VC'
        assert best_model.code.split('\n')[20] == '$THETA  1.15573 ; time_varying'
        assert best_model.code.split('\n')[26] == '$OMEGA  0.0396751 ; IIV_RUV1'


def test_ruvsearch_blq(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'ruvsearch').glob('mox3.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'moxo_simulated_resmod.csv', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'ruvsearch' / 'mytab', tmp_path)

        # Introduce 0 in CWRES to mimic rows BLQ
        with open('mytab') as f:
            mytab_new = f.read().replace('-2.4366E+00', '0.0000E+00')

        with open('mytab', 'w') as f:
            f.write(mytab_new)

        model = Model.parse_model('mox3.mod')
        results = read_modelfit_results('mox3.mod')

        model = transform_blq(model, method='m4', lloq=0.05)

        assert len(results.residuals.loc[results.residuals['CWRES'] == 0]) > 0

        res = run_tool(
            'ruvsearch',
            model=model,
            results=results,
            skip=['IIV_on_RUV', 'time_varying'],
            max_iter=1,
        )

        assert len(res.cwres_models) > 1
