import json
import os
import os.path
import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_time_after_dose
from pharmpy.tools.external.results import parse_modelfit_results
from pharmpy.workflows import (
    LocalDirectoryDatabase,
    LocalModelDirectoryDatabase,
    ModelDatabase,
    ModelEntry,
    NullModelDatabase,
)
from pharmpy.workflows.hashing import ModelHash


def test_base_class():
    with pytest.raises(TypeError):
        ModelDatabase()


def test_local_directory(tmp_path):
    with chdir(tmp_path):
        os.mkdir("database")
        db = LocalDirectoryDatabase("database")
        with open("file.txt", "w") as fh:
            print("Hello!", file=fh)
        db.store_local_file(None, "file.txt")
        with open("database/file.txt", "r") as fh:
            assert fh.read() == "Hello!\n"

        dirname = "doesnotexist"
        db = LocalDirectoryDatabase(dirname)
        assert os.path.isdir(dirname)


def test_null_database():
    db = NullModelDatabase(klr=123, f="oe")
    db.store_local_file("path", 34)


def test_store_model(tmp_path, load_model_for_test, testdata):
    sep = os.path.sep
    with chdir(tmp_path):
        datadir = testdata / 'nonmem'
        shutil.copy(datadir / 'pheno_real.mod', 'pheno_real.mod')
        shutil.copy(datadir / 'pheno.dta', 'pheno.dta')
        model = load_model_for_test("pheno_real.mod")
        # Remove TAD from statements because we will be adding TAD to data
        model = model.replace(statements=model.statements[2:])

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)

        with open("database/.datasets/data1.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2\n"

        with open("database/.datasets/data1.datainfo", "r") as fh:
            obj = json.load(fh)
            assert 'columns' in obj
            assert obj['path'] == 'data1.csv'

        h = ModelHash(model)
        with open(f"database/{h}/model.ctl", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}data1.csv IGNORE=@\n'

        run1 = model.replace(name="run1")
        db.store_model(run1)

        assert not (Path("database") / ".datasets" / "data2.csv").is_file()

        run2 = model.replace(name="run2")
        run2 = add_time_after_dose(run2)
        db.store_model(run2)

        with open("database/.datasets/data2.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2,TAD\n"

        with open("database/.datasets/data2.datainfo", "r") as fh:
            obj = json.load(fh)
            assert 'columns' in obj
            assert obj['path'] == 'data2.csv'

        h = ModelHash(run2)
        with open(f"database/{h}/model.ctl", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}data2.csv IGNORE=@\n'


def test_store_and_retrieve_model_entry(tmp_path, load_model_for_test, testdata):
    sep = os.path.sep
    with chdir(tmp_path):
        datadir = testdata / 'nonmem'
        for path in (testdata / 'nonmem').glob('pheno_real.*'):
            shutil.copy2(path, tmp_path)
        shutil.copy(datadir / 'pheno.dta', 'pheno.dta')
        model = load_model_for_test("pheno_real.mod")
        modelfit_results = parse_modelfit_results(model, "pheno_real.mod")
        model_entry = ModelEntry(
            model=model,
            modelfit_results=modelfit_results,
            log=modelfit_results.log,
        )

        db = LocalModelDirectoryDatabase("database")
        db.store_model_entry(model_entry)

        with open("database/.datasets/data1.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2\n"

        with open("database/.datasets/data1.datainfo", "r") as fh:
            obj = json.load(fh)
            assert 'columns' in obj
            assert obj['path'] == 'data1.csv'

        h = ModelHash(model_entry.model)
        with open(f"database/{h}/model.ctl", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}data1.csv IGNORE=@\n'

        model_entry_retrieve = db.retrieve_model_entry(model)

        assert model_entry_retrieve.model == model
        assert model_entry_retrieve.modelfit_results.ofv == modelfit_results.ofv


def test_list_all_files(tmp_path, load_example_model_for_test, testdata):
    with chdir(tmp_path):
        model = load_example_model_for_test("pheno")

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)
        assert set(db.list_all_files(model)) == {'model.ctl'}

        files_to_copy = {'pheno.lst', 'pheno.ext', 'pheno.phi'}
        for file_name in files_to_copy:
            path = testdata / 'nonmem' / file_name
            db.store_local_file(model, path)

        assert set(db.list_all_files(model)) == {'model.ctl', *files_to_copy}

        with db.snapshot(model) as snp:
            assert set(snp.list_all_files()) == {'model.ctl', *files_to_copy}


def test_retrieve_file(tmp_path, load_example_model_for_test, testdata):
    with chdir(tmp_path):
        model = load_example_model_for_test("pheno")

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)
        db.store_local_file(model, testdata / 'nonmem' / 'pheno.lst')

        assert not (tmp_path / 'pheno.lst').exists()
        db.retrieve_file(model, 'pheno.lst', tmp_path)
        assert (tmp_path / 'pheno.lst').exists()

        db.store_local_file(model, testdata / 'nonmem' / 'pheno.ext')
        with db.snapshot(model) as snp:
            snp.retrieve_file('pheno.ext', tmp_path)
        assert (tmp_path / 'pheno.ext').exists()

        with pytest.raises(FileNotFoundError):
            db.retrieve_file(model, 'x', tmp_path)

        with pytest.raises(ValueError):
            db.retrieve_file(model, 'pheno.lst', tmp_path)
        db.retrieve_file(model, 'pheno.lst', tmp_path, force=True)


def test_retrieve_all_files(tmp_path, load_example_model_for_test, testdata):
    with chdir(tmp_path):
        model = load_example_model_for_test("pheno")

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)

        files_to_copy = {'pheno.lst', 'pheno.ext', 'pheno.phi'}
        for file_name in files_to_copy:
            path = testdata / 'nonmem' / file_name
            db.store_local_file(model, path)

        os.mkdir(tmp_path / 'dir1')
        db.retrieve_all_files(model, tmp_path / 'dir1')
        assert all((tmp_path / 'dir1' / file).exists() for file in files_to_copy)

        os.mkdir(tmp_path / 'dir2')
        with db.snapshot(model) as snp:
            snp.retrieve_all_files(tmp_path / 'dir2')
        assert all((tmp_path / 'dir2' / file).exists() for file in files_to_copy)

        with pytest.raises(ValueError):
            db.retrieve_all_files(model, Path('x'))

        with open(tmp_path / 'file.txt', 'w'):
            pass
        with pytest.raises(ValueError):
            db.retrieve_all_files(model, tmp_path / 'file.txt')

        with pytest.raises(ValueError):
            db.retrieve_all_files(model, tmp_path / 'dir1')

        db.retrieve_all_files(model, tmp_path / 'dir1', force=True)
