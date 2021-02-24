import os

from pharmpy.execute.databases.local_directory import LocalDirectoryDatabase


def test_local_directory(fs):
    os.mkdir("database")
    db = LocalDirectoryDatabase("database")
    with open("file.txt", "w") as fh:
        print("Hello!", file=fh)
    db.store_local_file(None, "file.txt")
    with open("database/file.txt", "r") as fh:
        assert fh.read() == "Hello!\n"
