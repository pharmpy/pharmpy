import os.path
import shutil

from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.tools import read_modelfit_results, run_tool
from pharmpy.workflows import ModelDatabase


def test_allometry(tmp_path, testdata):
    with chdir(tmp_path):
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'sdtab1', tmp_path)

        model = Model.parse_model('pheno_real.mod')
        results = read_modelfit_results('pheno_real.mod')
        datainfo = model.datainfo.replace(path=tmp_path / 'pheno.dta')
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
                # NOTE skip date, time, description etc
                if line[:6] == '$DATA ':
                    assert line == f'$DATA ..{sep}.datasets{sep}input_model.csv IGNORE=@\n'
                    break
