from __future__ import annotations

from typing import Iterable, Sequence, Union

from pharmpy.internals.immutable import Immutable

from .absorption import Absorption
from .absorption import repr_many as absorption_repr_many
from .model_feature import ModelFeature


class ModelFeatures(Immutable):
    def __init__(self, features: tuple[ModelFeature, ...]):
        self._features = features

    @classmethod
    def create(cls, features: Iterable[ModelFeature]):
        if not isinstance(features, Iterable):
            raise TypeError(f'Type of `feature` must be an iterable: got {type(features)}')
        other_types = {type(f) for f in features if not isinstance(f, ModelFeature)}
        if other_types:
            raise TypeError(f'Incorrect types in `features`: got {sorted(other_types)}')

        feature_map = {'absorption': []}
        for feat in features:
            if isinstance(feat, Absorption):
                if feat not in feature_map['absorption']:
                    feature_map['absorption'].append(feat)
            else:
                raise NotImplementedError

        features = _flatten([sorted(value) for value in feature_map.values()])
        return cls(features=tuple(features))

    def replace(self, **kwargs) -> ModelFeatures:
        features = kwargs.get('features', self._features)
        return ModelFeatures.create(features=features)

    @property
    def features(self):
        return self._features

    @property
    def absorption(self):
        absorption_features = []
        for feature in self.features:
            if isinstance(feature, Absorption):
                absorption_features.append(feature)
        return ModelFeatures.create(absorption_features)

    def __add__(self, other: Union[ModelFeature, ModelFeatures]):
        if isinstance(other, ModelFeature):
            return ModelFeatures.create(features=self.features + (other,))
        elif isinstance(other, ModelFeatures):
            return ModelFeatures.create(features=self.features + other.features)
        elif isinstance(other, Sequence):
            return ModelFeatures.create(features=self.features + tuple(other))
        else:
            return NotImplemented

    def __radd__(self, other: Union[ModelFeature, ModelFeatures]):
        return self + other

    def __len__(self):
        return len(self.features)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ModelFeatures):
            return False
        if len(self) != len(other):
            return False
        return all(feat1 == feat2 for feat1, feat2 in zip(self.features, other.features))

    def __repr__(self):
        absorption_repr = absorption_repr_many(self.absorption.features)
        feature_repr = [absorption_repr]
        return ';'.join(feature_repr)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
