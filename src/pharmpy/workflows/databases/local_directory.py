import shutil
from pathlib import Path

from pharmpy.model_factory import Model

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
        if Path(source_path).is_file():
            shutil.copy2(source_path, self.path)


class LocalDirectoryDatabase(ModelDatabase):
    def __init__(self, path='.', file_extension='.mod'):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        self.path = path
        self.file_extension = file_extension

    def store_local_file(self, model, path):
        if Path(path).is_file():
            shutil.copy2(path, self.path)

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / filename
        model = Model(path)
        model.read_modelfit_results(path)
        return model


class LocalModelDirectoryDatabase(LocalDirectoryDatabase):
    def store_local_file(self, model, path):
        if Path(path).is_file():
            destination = self.path / model.name
            if not destination.is_dir():
                destination.mkdir(parents=True)
            shutil.copy2(path, destination)

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / filename
        model = Model(path)
        model.read_modelfit_results(self.path / name)
