import shutil
from pathlib import Path

from ..database import ModelDatabase, ToolDatabase


class LocalDirectoryToolDatabase(ToolDatabase):
    def __init__(self, toolname, path=None):
        if path is None:
            i = 1
            while True:
                name = f'{toolname}_dir{i}'
                if path is not None:
                    test_path = path / name
                else:
                    test_path = Path(name)
                if not test_path.exists():
                    path = test_path
                    break
                i += 1
        path = Path(path).resolve()
        path.mkdir(parents=True)
        self.path = path

        modeldb = LocalDirectoryDatabase(path / 'models')
        self.model_database = modeldb
        super().__init__(toolname)

    def store_local_file(self, source_path):
        shutil.copy2(source_path, self.path)


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
