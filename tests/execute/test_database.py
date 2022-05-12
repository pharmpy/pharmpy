import os
import os.path
import shutil
from pathlib import Path

import pytest

from pharmpy.modeling import add_time_after_dose, copy_model, read_model
from pharmpy.utils import TemporaryDirectoryChanger
from pharmpy.workflows import (
    LocalDirectoryDatabase,
    LocalModelDirectoryDatabase,
    ModelDatabase,
    NullModelDatabase,
    NullToolDatabase,
)


def test_base_class():
    with pytest.raises(TypeError):
        ModelDatabase()


def test_local_directory(tmp_path):
    with TemporaryDirectoryChanger(tmp_path):
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
    db = NullToolDatabase("any", sl1=23, model=45, opr=12, dummy="some dummy kwargs")
    db.store_local_file("path")
    db = NullModelDatabase(klr=123, f="oe")
    db.store_local_file("path", 34)


def test_store_model(tmp_path, testdata):
    sep = os.path.sep
    with TemporaryDirectoryChanger(tmp_path):
        datadir = testdata / 'nonmem'
        shutil.copy(datadir / 'pheno_real.mod', 'pheno_real.mod')
        shutil.copy(datadir / 'pheno.dta', 'pheno.dta')
        model = read_model("pheno_real.mod")

        db = LocalModelDirectoryDatabase("database")
        db.store_model(model)

        with open("database/.datasets/pheno_real.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2\n"

        with open("database/.datasets/pheno_real.datainfo", "r") as fh:
            line = fh.readline()
            assert line.startswith('{"columns": [{"name": ')

        with open("database/pheno_real/pheno_real.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line.endswith(f'{sep}database{sep}.datasets{sep}pheno_real.csv IGNORE=@\n')

        run1 = copy_model(model, name="run1")
        db.store_model(run1)

        assert not (Path("database") / ".datasets" / "run1.csv").is_file()

        with open("database/run1/run1.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line.endswith(f'{sep}database{sep}.datasets{sep}pheno_real.csv IGNORE=@\n')

        run2 = copy_model(model, name="run2")
        add_time_after_dose(run2)
        db.store_model(run2)

        with open("database/.datasets/run2.csv", "r") as fh:
            line = fh.readline()
            assert line == "ID,TIME,AMT,WGT,APGR,DV,FA1,FA2,TAD\n"

        with open("database/.datasets/run2.datainfo", "r") as fh:
            line = fh.readline()
            assert line.startswith('{"columns": [{"name": ')

        with open("database/run2/run2.mod", "r") as fh:
            line = fh.readline()
            assert line == "$PROBLEM PHENOBARB SIMPLE MODEL\n"
            line = fh.readline()
            assert line.endswith(f'{sep}database{sep}.datasets{sep}run2.csv IGNORE=@\n')
