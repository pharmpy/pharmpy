from os.path import normpath, relpath
from pathlib import Path


def path_relative_to(root: Path, path: Path) -> Path:
    return Path(normpath(relpath(str(path), start=str(root.parent))))
