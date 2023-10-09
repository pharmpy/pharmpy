from ...workflows.model_entry import ModelEntry
from .baseclass import NonTransactionalModelDatabase


class NullModelDatabase(NonTransactionalModelDatabase):
    """Dummy model database implementation

    No operation does anything. This database can be used if no storing of files
    is desirable.
    """

    def __init__(self, **kwargs):
        pass

    def store_model(self, model):
        pass

    def store_local_file(self, model, path, new_filename=None):
        pass

    def store_metadata(self, model, metadata):
        pass

    def store_modelfit_results(self, model):
        pass

    def store_model_entry(self, model_entry: ModelEntry) -> None:
        pass

    def retrieve_file(self, model_name, filename):
        pass

    def retrieve_local_files(self, name, destination_path):
        pass

    def retrieve_model(self, name):
        pass

    def retrieve_modelfit_results(self, name):
        pass

    def retrieve_model_entry(self, name):
        pass
