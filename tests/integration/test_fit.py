import filecmp
import shutil

import pytest

import pharmpy.modeling as modeling
from pharmpy.config import site_config_path, user_config_path
from pharmpy.deps import pandas as pd
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model
from pharmpy.modeling import filter_dataset
from pharmpy.tools import fit
from pharmpy.tools.external.nlmixr import verification as nlmixr_verification
from pharmpy.tools.external.nonmem import conf
from pharmpy.tools.external.nonmem.run import execute_model as nonmem_execute_model
from pharmpy.tools.external.rxode import verification as rxode_verification
from pharmpy.tools.run import run_tool
from pharmpy.workflows import LocalDirectoryContext, ModelEntry


def test_configuration():
    print("User config dir:", user_config_path())
    print("Site config dir:", site_config_path())
    print("Default NONMEM path:", conf.default_nonmem_path)
    assert (conf.default_nonmem_path / 'license' / 'nonmem.lic').is_file()


def test_fit_single(tmp_path, model_count, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = fit(model, validate_dataset=True)
        rundir = tmp_path / 'modelfit1'
        assert res.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
        assert model_count(rundir) == 1
        assert (rundir / 'models' / 'pheno' / '.pharmpy').exists()
        assert not [
            path.name for path in (rundir / 'models' / 'pheno').iterdir() if 'contr' in path.name
        ]
        original_dataset_path = tmp_path / 'pheno.dta'
        context_dataset_path = rundir / '.modeldb' / '.datasets' / 'pheno.dta'
        assert filecmp.cmp(original_dataset_path, context_dataset_path)


def test_fit_multiple(tmp_path, model_count, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno_1.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno_1.dta')
        model_1 = Model.parse_model('pheno_1.mod')
        df = pd.read_table(tmp_path / 'pheno_1.dta', sep=r'\s+', header=0)
        # Mimic NONMEM dataset parsing
        df.index = range(1, len(df) + 1)
        df['ID'] = df['ID'].astype('int32')
        model_1 = model_1.replace(
            dataset=df, datainfo=model_1.datainfo.replace(path=tmp_path / 'pheno_1.dta')
        )
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno_2.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno_2.dta')
        model_2 = Model.parse_model('pheno_2.mod')
        model_2 = model_2.replace(
            dataset=df, datainfo=model_2.datainfo.replace(path=tmp_path / 'pheno_2.dta')
        )
        res1, res2 = fit([model_1, model_2], validate_dataset=True)
        rundir = tmp_path / 'modelfit1'
        assert res1.ofv == pytest.approx(730.8947268137308)
        assert res2.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
        assert model_count(rundir) == 2


def test_fit_copy(tmp_path, model_count, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path / 'pheno.mod')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path / 'pheno.dta')

        model_1 = Model.parse_model('pheno.mod')
        model_1 = model_1.replace(datainfo=model_1.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res1 = fit(model_1)

        rundir_1 = tmp_path / 'modelfit1'
        assert rundir_1.is_dir()
        assert model_count(rundir_1) == 1

        model_2 = model_1.replace(name='pheno_copy')
        model_2 = modeling.set_initial_estimates(model_2, res1.parameter_estimates)
        res2 = fit(model_2)

        rundir_2 = tmp_path / 'modelfit1'
        assert rundir_2.is_dir()
        assert model_count(rundir_2) == 1

        assert res1.ofv != res2.ofv


def test_fit_nlmixr(tmp_path, testdata):
    from pharmpy.tools.external.nlmixr import conf

    if str(conf.rpath) == '.':
        pytest.skip("No R selected in conf. Skipping nlmixr tests")
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        model = modeling.convert_model(model, 'nlmixr')
        res = fit(model, esttool='nlmixr')
        assert res.ofv == pytest.approx(
            1672.1592
        )  # Significantly higher than NONMEM but predictions values are similar (see test_verification)
        assert res.parameter_estimates['TVCL'] == pytest.approx(0.23364, abs=1e-4)


def test_fit_dummy(tmp_path, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = fit(model, esttool='dummy')
        rundir = tmp_path / 'modelfit1'
        assert res.ofv == -13.40677997809423
        assert rundir.is_dir()


def test_verification_nlmixr(tmp_path, testdata):
    from pharmpy.tools.external.nlmixr import conf

    if str(conf.rpath) == '.':
        pytest.skip("No R selected in conf. Skipping nlmixr tests")
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_real.ctl')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno_real.ctl')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        assert nlmixr_verification(model)


def test_verification_rxode(tmp_path, testdata):
    from pharmpy.tools.external.nlmixr import conf

    if str(conf.rpath) == '.':
        pytest.skip("No R selected in conf. Skipping rxode tests")
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno_real.mod', tmp_path / 'pheno_real.ctl')
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno_real.ctl')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        assert rxode_verification(model)


def test_execute_model_nonmem(tmp_path, testdata):
    with chdir(tmp_path):
        datadir = testdata / 'nonmem'
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy(datadir / 'pheno.dta', 'pheno.dta')
        model = Model.parse_model('pheno_real.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))

        model_entry = ModelEntry.create(
            model=model,
            modelfit_results=None,
            log=None,
        )

        db = LocalDirectoryContext('db_model')

        model_entry = nonmem_execute_model(model_entry, db)

        assert isinstance(model_entry, ModelEntry)
        assert model_entry.modelfit_results
        assert (db.path / 'models' / 'pheno_real').is_dir()


def test_fit_ignore_statements(tmp_path, model_count, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox2.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.csv', tmp_path)
        model = Model.parse_model('mox2.mod')
        model = model.replace(
            datainfo=model.datainfo.replace(path=tmp_path / 'mox_simulated_normal.csv')
        )
        model = filter_dataset(model, 'VISI == 3')
        res = fit(model, validate_dataset=True)
        rundir = tmp_path / 'modelfit1'
        assert res.ofv == pytest.approx(-706.1332300694054)
        assert rundir.is_dir()
        assert model_count(rundir) == 1
        assert (rundir / 'models' / 'mox2' / '.pharmpy').exists()
        assert not [
            path.name for path in (rundir / 'models' / 'mox2').iterdir() if 'contr' in path.name
        ]
        original_dataset_path = tmp_path / 'mox_simulated_normal.csv'
        context_dataset_path = rundir / '.modeldb' / '.datasets' / 'mox_simulated_normal.csv'
        assert filecmp.cmp(original_dataset_path, context_dataset_path)


def test_fit_use_pharmpy_dataset(tmp_path, model_count, testdata):
    with chdir(tmp_path):
        shutil.copy2(testdata / 'nonmem' / 'pheno.mod', tmp_path)
        shutil.copy2(testdata / 'nonmem' / 'pheno.dta', tmp_path)
        model = Model.parse_model('pheno.mod')
        model = model.replace(datainfo=model.datainfo.replace(path=tmp_path / 'pheno.dta'))
        res = run_tool('modelfit', model, always_create_new_dataset_file=True)
        rundir = tmp_path / 'modelfit1'
        assert res.ofv == pytest.approx(730.8947268137308)
        assert rundir.is_dir()
        assert model_count(rundir) == 1
        assert (rundir / 'models' / 'pheno' / '.pharmpy').exists()
        assert not [
            path.name for path in (rundir / 'models' / 'pheno').iterdir() if 'contr' in path.name
        ]
        assert not (rundir / '.modeldb' / '.datasets' / 'pheno.dta').is_file()
        assert (rundir / '.modeldb' / '.datasets' / 'data1.csv').is_file()
