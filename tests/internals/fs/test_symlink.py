from pharmpy.internals.fs.symlink import create_directory_symlink


def test_create_directory_symlink(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    link_dir = tmp_path / "link"

    create_directory_symlink(link_dir, target_dir)

    assert link_dir.is_dir()
    assert target_dir.is_dir()

    test_file_path = target_dir / "testfile"
    link_file_path = link_dir / "testfile"
    assert not test_file_path.exists()
    assert not link_file_path.exists()

    epictetus_quote = "No great thing is created suddenly."
    with open(test_file_path, 'w') as fh:
        fh.write(epictetus_quote)

    with open(link_file_path, 'r') as fh:
        text = fh.read()

    assert text == epictetus_quote
