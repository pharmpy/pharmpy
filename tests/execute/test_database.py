import os
import os.path

import pytest

from pharmpy.workflows import (
    LocalDirectoryDatabase,
    ModelDatabase,
    NullModelDatabase,
    NullToolDatabase,
)


def test_base_class():
    with pytest.raises(TypeError):
        ModelDatabase()


def test_local_directory(fs):
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
