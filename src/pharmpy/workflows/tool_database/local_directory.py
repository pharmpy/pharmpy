import shutil
from pathlib import Path

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
    """

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

    def store_results(self, res):
        res.to_json(path=self.path / 'results.json')
        res.to_csv(path=self.path / 'results.csv')
