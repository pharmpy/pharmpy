import shutil

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import fix_parameters, remove_parameter_uncertainty_step
from pharmpy.tools import fit, run_iovsearch


def test_default_mox2(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_iovsearch(
            model=start_modelres[0], results=start_modelres[1], column='VISI', rank_type='bic'
        )
        rundir = tmp_path / 'iovsearch1'
        assert model_count(rundir) == 10

        assert res.final_model.name == 'iovsearch_run7'


def test_ignore_fixed_iiv(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        start_model = fix_parameters(start_modelres[0], 'IIV_CL')
        res = run_iovsearch(model=start_model, results=start_modelres[1], column='VISI')
        assert len(res.summary_models) == 5


def test_rank_type_ofv_mox2(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_iovsearch(
            model=start_modelres[0], results=start_modelres[1], column='VISI', rank_type='ofv'
        )
        rundir = tmp_path / 'iovsearch1'
        assert model_count(rundir) == 10

        assert res.final_model.name == 'iovsearch_run7'


def test_default_mox1(tmp_path, model_count, testdata):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox1.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_log.csv', tmp_path)
    with chdir(tmp_path):
        start_model = Model.parse_model('mox1.mod')
        start_res = fit(start_model)
        res = run_iovsearch(model=start_model, results=start_res, column='VISI')
        rundir = tmp_path / 'iovsearch1'
        assert model_count(rundir) == 9

        assert res.final_model.name == 'input'


@pytest.mark.parametrize(
    'kwargs, no_of_candidate_models, best_model',
    [
        (dict(), 9, 'iovsearch_run8'),
        ({'rank_type': 'mbic', 'E': 1.0}, 9, 'iovsearch_run8'),
        (
            {
                'parameter_uncertainty_method': 'SANDWICH',
                'strictness': 'minimization_successful and rse <= 0.5',
            },
            9,
            'iovsearch_run8',
        ),
    ],
)
def test_iovsearch_dummy(
    tmp_path, model_count, testdata, kwargs, no_of_candidate_models, best_model
):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox1.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_log.csv', tmp_path)
    with chdir(tmp_path):
        start_model = Model.parse_model('mox1.mod')
        start_model = remove_parameter_uncertainty_step(start_model)
        start_res = fit(start_model, esttool='dummy')
        res = run_iovsearch(
            model=start_model, results=start_res, column='VISI', esttool='dummy', **kwargs
        )

        assert res.final_model.name == best_model

        rundir = tmp_path / 'iovsearch1'
        assert model_count(rundir) == no_of_candidate_models + 1
        assert (rundir / 'results.json').exists()
        assert (rundir / 'results.csv').exists()
        assert (rundir / 'metadata.json').exists()
        assert (rundir / 'models' / 'iovsearch_run1' / 'model_results.json').exists()
        assert not (rundir / 'models' / 'iovsearch_run1' / 'model.lst').exists()
