import json
import shutil
from itertools import count
from pathlib import Path

from pharmpy.internals.fs.path import path_absolute

from ..model_database import LocalModelDirectoryDatabase
from .baseclass import ToolDatabase


class LocalDirectoryToolDatabase(ToolDatabase):
    """ToolDatabase in a local directory

    Parameters
    ----------
    toolname : str
        Name of the tool
    path : str or Path
        Path to directory. Will be created if it does not exist.
    exist_ok : bool
        Whether to allow using an existing database.
    """

    def __init__(self, toolname, path=None, exist_ok=False):
        if path is None:
            for i in count(1):
                name = f'{toolname}_dir{i}'
                path = Path(name)
                try:
                    path.mkdir(parents=True)
                    break
                except FileExistsError:
                    pass
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=exist_ok)

        assert path is not None
        self.path = path_absolute(path)

        modeldb = LocalModelDirectoryDatabase(self.path / 'models')
        self.model_database = modeldb

        super().__init__(toolname)

    def to_dict(self):
        return {'toolname': self.toolname, 'path': str(self.path)}

    @classmethod
    def from_dict(cls, d):
        return cls(**d, exist_ok=True)

    def store_local_file(self, source_path):
        if Path(source_path).is_file():
            shutil.copy2(source_path, self.path)

    def store_results(self, res):
        res.to_json(path=self.path / 'results.json')
        res.to_csv(path=self.path / 'results.csv')

    def store_metadata(self, metadata):
        path = self.path / 'metadata.json'
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def read_metadata(self):
        path = self.path / 'metadata.json'
        with open(path, 'r') as f:
            metadata = json.load(f)
        return metadata
