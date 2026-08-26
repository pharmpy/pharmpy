from pharmpy.workflows import LocalDirectoryProject


def test_init(tmp_path):
    proj = LocalDirectoryProject('myproject', tmp_path)
    assert proj.path == tmp_path / 'myproject'
    assert (proj.path / '.modeldb').is_dir()
    assert repr(proj) == f'<Local directory project at {tmp_path / "myproject"}>'

    proj2 = LocalDirectoryProject('myproject', tmp_path)
    assert proj2.path == proj.path


def test_init_already_existing_dir(tmp_path):
    proj_path = tmp_path / 'myproject'
    assert not proj_path.is_dir()
    proj_path.mkdir()
    assert not (proj_path / '.modeldb').is_dir()
    proj = LocalDirectoryProject('myproject', tmp_path)
    assert proj.path == tmp_path / 'myproject'
    assert (proj_path / '.modeldb').is_dir()
