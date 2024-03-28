import os.path
import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, run_tool
from pharmpy.workflows import LocalDirectoryContext


def test_allometry(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem' / 'models').glob('pheno5.*'):
            shutil.copy2(path, tmp_path)

        model = Model.parse_model('pheno5.mod')
        results = read_modelfit_results('pheno5.mod')
        datainfo = model.datainfo.replace(path=tmp_path / 'pheno5.csv')
        model = model.replace(datainfo=datainfo)
        res = run_tool('allometry', model=model, results=results, allometric_variable='WGT')
        assert len(res.summary_models) == 2

        context = LocalDirectoryContext("allometry1")
        sep = os.path.sep
        model_name = 'scaled_model'
        assert str(context.retrieve_model_entry(model_name).model.datainfo.path).endswith(
            f'{sep}allometry1{sep}.modeldb{sep}.datasets{sep}data1.csv'
        )
        path = context.model_database.retrieve_file(context.retrieve_key(model_name), 'model.lst')
        with open(path, 'r') as fh:
            while line := fh.readline():
                # NOTE: Skip date, time, description etc
                if line[:6] == '$DATA ':
                    assert line == f'$DATA ..{sep}.datasets{sep}data1.csv IGNORE=@\n'
                    break
