from pathlib import Path

import pharmpy.utils as utils


def test_tempdir():
    with utils.TemporaryDirectory() as td:
        path = Path(td)
        assert path.is_dir()
    assert not path.is_dir()
