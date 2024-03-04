import shutil
import sys

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import fix_parameters
from pharmpy.tools import fit, read_modelfit_results, run_iovsearch

tflite_condition = (
    sys.version_info >= (3, 12)
    and sys.platform == 'win32'
    or sys.version_info >= (3, 12)
    and sys.platform == 'darwin'
)


@pytest.mark.slow
def test_default_mox2(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_iovsearch(
            'VISI', rank_type='bic', results=start_modelres[1], model=start_modelres[0]
        )
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 8

        assert res.final_model.name == 'iovsearch_run7'


@pytest.mark.slow
def test_ignore_fixed_iiv(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        start_model = fix_parameters(start_modelres[0], 'IIV_CL')
        res = run_iovsearch('VISI', results=start_modelres[1], model=start_model)
        assert len(res.summary_models) == 5


@pytest.mark.slow
def test_rank_type_ofv_mox2(tmp_path, model_count, start_modelres):
    with chdir(tmp_path):
        res = run_iovsearch(
            'VISI', results=start_modelres[1], model=start_modelres[0], rank_type='ofv'
        )
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 8

        assert res.final_model.name == 'iovsearch_run7'


@pytest.mark.slow
def test_default_mox1(tmp_path, model_count, testdata):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox1.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_log.csv', tmp_path)
    with chdir(tmp_path):
        start_model = Model.parse_model('mox1.mod')
        start_res = fit(start_model)
        res = run_iovsearch('VISI', results=start_res, model=start_model)
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 7

        assert res.final_model.name == start_model.name


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_default_mox2_dummy(tmp_path, load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'models' / 'mox2.mod')
    with chdir(tmp_path):
        run_iovsearch(
            'VISI', rank_type='ofv', results=results, model=model, estimation_tool='dummy'
        )
