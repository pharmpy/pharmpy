import os.path
import shutil
import sys

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, run_tool
from pharmpy.workflows import ModelDatabase

tflite_condition = (
    sys.version_info >= (3, 12)
    and sys.platform == 'win32'
    or sys.version_info >= (3, 12)
    and sys.platform == 'darwin'
)


@pytest.mark.slow
def test_allometry(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'models').glob('pheno5.*'):
            shutil.copy2(path, tmp_path)
        # shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        # shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.parse_model('pheno5.mod')
        results = read_modelfit_results('pheno5.mod')
        datainfo = model.datainfo.replace(path=tmp_path / 'pheno5.csv')
        model = model.replace(datainfo=datainfo)
        res = run_tool('allometry', model=model, results=results, allometric_variable='WGT')
        assert len(res.summary_models) == 2

        db: ModelDatabase = res.tool_database.model_database
        sep = os.path.sep
        model_name = 'scaled_model'
        assert str(db.retrieve_model(model_name).datainfo.path).endswith(
            f'{sep}allometry_dir1{sep}models{sep}.datasets{sep}input_model.csv'
        )
        path = db.retrieve_file(model_name, f'{model_name}.lst')
        with open(path, 'r') as fh:
            while line := fh.readline():
                # NOTE: Skip date, time, description etc
                if line[:6] == '$DATA ':
                    assert line == f'$DATA ..{sep}.datasets{sep}input_model.csv IGNORE=@\n'
                    break


@pytest.mark.skipif(tflite_condition, reason="Skipping tests requiring tflite for Python 3.12")
def test_run_allometry(tmp_path, load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'pheno5.mod')
    results = read_modelfit_results(testdata / 'nonmem' / 'models' / 'pheno5.mod')
    with chdir(tmp_path):
        res = run_tool(
            'allometry',
            model=model,
            results=results,
            allometric_variable='WGT',
            estimation_tool='dummy',
        )
        assert len(res.summary_models) == 2
