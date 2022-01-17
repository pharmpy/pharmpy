from abc import ABC, abstractmethod


class ModelDatabase(ABC):
    """Database for models and results of model runs

    primary key is the model name
    """

    @abstractmethod
    def store_local_file(self, model, path):
        """Store a file from the local machine"""
        pass

    @abstractmethod
    def retrieve_local_files(self, name, destination_path):
        pass

    @abstractmethod
    def retrieve_file(self, model_name, filename):
        pass

    @abstractmethod
    def get_model(self, name):
        pass
