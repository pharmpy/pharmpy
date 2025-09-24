from abc import abstractmethod

from pharmpy.internals.immutable import Immutable


class ModelFeature(Immutable):
    @property
    @abstractmethod
    def args(self):
        pass

    @abstractmethod
    def expand(self, model):
        pass

    @staticmethod
    @abstractmethod
    def repr_many():
        pass
