from dataclasses import dataclass

from pharmpy.internals.fs.cwd import chdir
from pharmpy.workflows import LocalDirectoryToolDatabase, NullToolDatabase


@dataclass(frozen=True)
class Bar:
    z: float


@dataclass(frozen=True)
class Foo:
    x: int
    y: Bar


def test_null_tool_database():
    db = NullToolDatabase("any", sl1=23, model=45, opr=12, dummy="some dummy kwargs")
    db.store_local_file("path")


def test_metadata_dataclasses_round_trip(tmp_path):
    with chdir(tmp_path):
        db = LocalDirectoryToolDatabase('test_tool')
        expected = Foo(1, Bar(0.5))
        db.store_metadata(expected)
        result = db.read_metadata()
        assert result == expected
