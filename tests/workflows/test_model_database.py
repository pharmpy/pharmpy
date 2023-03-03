import json
import os
import os.path
import shutil
from pathlib import Path

import pytest

from pharmpy.internals.fs.cwd import chdir
from pharmpy.modeling import add_time_after_dose
from pharmpy.workflows import (
    LocalDirectoryDatabase,
    LocalModelDirectoryDatabase,
    ModelDatabase,
    NullModelDatabase,
)


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

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)

        with open("database/.datasets/pheno_real.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2\n"

        with open("database/.datasets/pheno_real.datainfo", "r") as fh:
            obj = json.load(fh)
            assert 'columns' in obj
            assert obj['path'] == 'pheno_real.csv'

        with open("database/pheno_real/pheno_real.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}pheno_real.csv IGNORE=@\n'

        run1 = model.replace(name="run1")
        db.store_model(run1)

        assert not (Path("database") / ".datasets" / "run1.csv").is_file()

        with open("database/run1/run1.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}pheno_real.csv IGNORE=@\n'

        run2 = model.replace(name="run2")
        run2 = add_time_after_dose(run2)
        db.store_model(run2)

        with open("database/.datasets/run2.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2,TAD\n"

        with open("database/.datasets/run2.datainfo", "r") as fh:
            obj = json.load(fh)
            assert 'columns' in obj
            assert obj['path'] == 'run2.csv'

        with open("database/run2/run2.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line == f'$DATA ..{sep}.datasets{sep}run2.csv IGNORE=@\n'
