from pathlib import Path

from pharmpy.internals.fs.tmp import TemporaryDirectory


def test_tempdir():
    with TemporaryDirectory() as td:
        path = Path(td)
        assert path.is_dir()
    assert not path.is_dir()
