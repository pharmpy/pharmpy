import shutil
from os import stat
from pathlib import Path

from pharmpy.model_factory import Model

from ..database import ModelDatabase, ToolDatabase


class LocalDirectoryToolDatabase(ToolDatabase):
    def __init__(self, toolname, path=None):
        if path is None:
            i = 1
            while True:
                name = f'{toolname}_dir{i}'
                path = Path(name)
                if not path.exists():
                    break
                i += 1
        path = Path(path)
        path.mkdir(parents=True)
        self.path = path.resolve()

        modeldb = LocalModelDirectoryDatabase(self.path / 'models')
        self.model_database = modeldb
        super().__init__(toolname)

    def store_local_file(self, source_path):
        if Path(source_path).is_file():
            shutil.copy2(source_path, self.path)


class LocalDirectoryDatabase(ModelDatabase):
    # Files are all stored in the same directory
    # Assuming filenames connected to a model are named modelname + extension
    def __init__(self, path='.', file_extension='.mod'):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        self.path = path.resolve()
        self.file_extension = file_extension

    def store_local_file(self, model, path):
        if Path(path).is_file():
            shutil.copy2(path, self.path)

    def retrieve_local_files(self, name, destination_path):
        # Retrieve all files stored for one model
        files = self.path.glob(f'{name}.*')
        for f in files:
            shutil.copy2(f, destination_path)

    def retrieve_file(self, model_name, filename):
        # Return path to file
        path = self.path / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {model_name}")

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / filename
        try:
            model = Model(path)
        except FileNotFoundError:
            raise KeyError('Model cannot be found in database')
        model.database = self
        model.read_modelfit_results()
        return model

    def __repr__(self):
        return f"LocalDirectoryDatabase({self.path})"


class LocalModelDirectoryDatabase(LocalDirectoryDatabase):
    def store_local_file(self, model, path):
        if Path(path).is_file():
            destination = self.path / model.name
            if not destination.is_dir():
                destination.mkdir(parents=True)
            shutil.copy2(path, destination)

    def retrieve_local_files(self, name, destination_path):
        path = self.path / name
        files = path.glob('*')
        for f in files:
            shutil.copy2(f, destination_path)

    def retrieve_file(self, model_name, filename):
        # Return path to file
        path = self.path / model_name / filename
        if path.is_file() and stat(path).st_size > 0:
            return path
        else:
            raise FileNotFoundError(f"Cannot retrieve {filename} for {model_name}")

    def get_model(self, name):
        filename = name + self.file_extension
        path = self.path / name / filename
        model = Model(path)
        model.database = self
        model.read_modelfit_results()
        return model

    def __repr__(self):
        return f"LocalModelDirectoryDatabase({self.path})"
