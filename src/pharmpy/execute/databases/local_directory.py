import shutil
from pathlib import Path

from ..database import ModelDatabase


class LocalDirectoryDatabase(ModelDatabase):
    def __init__(self, path='.'):
        path = Path(path).resolve()
        if not path.exists():
            path.mkdir(parents=True)
        self.path = path

    def store_local_file(self, model, path):
        shutil.copy2(path, self.path)


class LocalModelDirectoryDatabase(ModelDatabase):
    def store_local_file(self, model, path):
        shutil.copy2(path, model.source.path.parent)
