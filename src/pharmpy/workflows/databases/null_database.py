from ..database import ModelDatabase, ToolDatabase


class NullToolDatabase(ToolDatabase):
    def __init__(self, toolname, **kwargs):
        self.model_database = NullModelDatabase()
        super().__init__(toolname)

    def store_local_file(self, source_path):
        pass


class NullModelDatabase(ModelDatabase):
    def __init__(self, **kwargs):
        pass

    def store_local_file(self, model, path):
        pass

    def retrieve_file(self, model_name, filename):
        pass

    def get_model(self, name):
        pass
