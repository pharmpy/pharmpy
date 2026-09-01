from pathlib import Path
from typing import Optional, Union

from pharmpy.internals.fs.path import path_absolute

from ..model_database import LocalModelDirectoryDatabase
from .baseclass import Project


class LocalDirectoryProject(Project):
    def __init__(self, name: str, ref: Optional[Union[str, Path]] = None):
        if ref is None:
            ref = str(Path.cwd())
        else:
            ref = str(ref)
        path = Path(ref) / name

        self._init_path(path)
        self._init_model_database()

        super().__init__(name, ref)

    def __repr__(self) -> str:
        return f"<Local directory project at {self.path}>"

    def _init_path(self, path):
        self.path = path_absolute(path)
        if not self.path.is_dir():
            self.path.mkdir(parents=True)

    def _init_model_database(self):
        self.model_database = LocalModelDirectoryDatabase(self.path / '.modeldb')

    def get_context_ref(self, ref: Optional[str]) -> str:
        if ref is not None:
            new_ref = self.path / ref
        else:
            new_ref = self.path
        return str(new_ref)
