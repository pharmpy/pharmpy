from ..model_database import NullModelDatabase
from .baseclass import ToolDatabase


class NullToolDatabase(ToolDatabase):
    def __init__(self, toolname, **kwargs):
        self.model_database = NullModelDatabase()
        super().__init__(toolname)

    def store_local_file(self, source_path):
        pass
