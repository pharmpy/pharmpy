import builtins

from .model_feature import ModelFeature


class MutexFeature(ModelFeature):
    def __init__(self, type):
        self._type = type

    @classmethod
    def create(cls, type):
        if not isinstance(type, str):
            raise TypeError(f'Type of `type` must be a string: got {builtins.type(type)}')

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

        order = self.__class__.order

        def _get_complexity(obj):
            return order.get(obj.type)

        return _get_complexity(self) < _get_complexity(other)

    def __repr__(self):
        return f'{self.__class__.__name__.upper()}({self.type})'

    @staticmethod
    def repr_many(features):
        if len(features) == 1:
            return repr(features[0])
        features = sorted(features)
        class_name = features[0].__class__.__name__.upper()
        return f"{class_name}([{','.join(feat.type for feat in features)}])"
