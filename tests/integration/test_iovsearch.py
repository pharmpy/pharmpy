import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import fit, run_iovsearch


def test_default_mox2(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_iovsearch('VISI', results=start_model.modelfit_results, model=start_model)
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 8

        assert res.final_model_name == 'iovsearch_run7'


def test_rank_type_ofv_mox2(tmp_path, model_count, start_model):
    with chdir(tmp_path):
        res = run_iovsearch(
            'VISI', results=start_model.modelfit_results, model=start_model, rank_type='ofv'
        )
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 8

        assert res.final_model_name == 'iovsearch_run7'


def test_default_mox1(tmp_path, model_count, testdata):
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox1.mod', tmp_path)
    shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_log.csv', tmp_path)
    with chdir(tmp_path):
        start_model = Model.parse_model('mox1.mod')
        start_res = fit(start_model)
        start_model = start_model.replace(modelfit_results=start_res)
        res = run_iovsearch('VISI', results=start_res, model=start_model)
        rundir = tmp_path / 'iovsearch_dir1'
        assert model_count(rundir) == 7

        assert res.final_model_name == start_model.name
