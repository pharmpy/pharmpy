from os.path import normpath, relpath
from pathlib import Path
from typing import Union


def path_relative_to(root: Path, path: Path) -> Path:
    # NOTE A ValueError will be raised on Windows if path and root are on
    # different drives.
    return Path(normpath(relpath(str(path_absolute(path)), start=str(path_absolute(root)))))


def path_absolute(path: Path) -> Path:
    # NOTE This makes the path absolute without resolving symlinks
    new_path = (
        Path(normpath(str(path))) if path.is_absolute() else Path(normpath(str(Path.cwd() / path)))
    )
    assert new_path.is_absolute()
    return new_path


def normalize_user_given_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path.expanduser()
