from pharmpy.workflows import RunDirectory


def test_RunDirectory_temporary():
    rd = RunDirectory()
    path = rd.path
    assert path.is_dir()

    del rd
    assert not path.exists()


def test_RunDirectory_persistent():
    rd = RunDirectory('.')
    path = rd.path
    assert path.is_dir()

    contents = list(rd.path.iterdir())
    assert not contents
    rd.cleanlevel = 1
    rd.clean_config(level=1, patterns=['*.temp'], rm_dirs=False)
    to_remove = path / 'some.temp'
    to_keep = path / 'save.txt'
    open(str(to_remove), 'a').close()
    open(str(to_keep), 'a').close()

    rd.cleanup()
    assert path.exists()
    assert to_keep.exists()
    assert not to_remove.exists()

    del rd
    assert path.exists()
    assert to_keep.exists()
    (path / 'save.txt').unlink()
    path.rmdir()

    assert not path.exists()
