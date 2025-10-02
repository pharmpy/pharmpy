from abc import abstractmethod

from .model_feature import ModelFeature


class MutexFeature(ModelFeature):
    def __init__(self, type):
        self._type = type

    @classmethod
    @abstractmethod
    def create(cls, type):
        pass

    def replace(self, **kwargs):
        type = kwargs.get('type', self.type)
        return self.__class__.create(type=type)

    @property
    def type(self):
        return self._type

    @property
    def args(self):
        return (self.type,)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, self.__class__):
            return False
        return self.type == other.type

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            return False

        return self.get_complexity() < other.get_complexity()

    def __repr__(self):
        return f'{self.__class__.__name__.upper()}({self.type})'

    @staticmethod
    def repr_many(mfl):
        features = sorted(mfl.features)
        if len(features) == 1:
            return repr(features[0])
        class_name = features[0].__class__.__name__.upper()
        return f"{class_name}([{','.join(feat.type for feat in features)}])"
