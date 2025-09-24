from __future__ import annotations

from typing import Iterable, Sequence, Union

from pharmpy.internals.immutable import Immutable

from .features import Absorption, ModelFeature, Peripherals, Transits
from .features.absorption import repr_many as absorption_repr_many
from .features.peripherals import repr_many as peripherals_repr_many
from .features.transits import repr_many as transits_repr_many


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

        type_to_group = {Absorption: 'absorption', Transits: 'transits', Peripherals: 'peripherals'}
        grouped_features = {group: [] for group in type_to_group.values()}
        for feature in features:
            group = type_to_group.get(type(feature))
            if group is None:
                raise NotImplementedError
            if feature not in grouped_features[group]:
                grouped_features[group].append(feature)

        features = _flatten([sorted(value) for value in grouped_features.values()])
        return cls(features=tuple(features))

    def replace(self, **kwargs) -> ModelFeatures:
        features = kwargs.get('features', self._features)
        return ModelFeatures.create(features=features)

    @property
    def features(self):
        return self._features

    @property
    def absorption(self):
        features = self._filter_by_type(Absorption)
        return ModelFeatures.create(features)

    @property
    def transits(self):
        features = self._filter_by_type(Transits)
        return ModelFeatures.create(features)

    @property
    def peripherals(self):
        features = self._filter_by_type(Peripherals)
        return ModelFeatures.create(features)

    def _filter_by_type(self, type):
        features = []
        for feature in self.features:
            if isinstance(feature, type):
                features.append(feature)
        return features

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
        feature_repr = []
        if self.absorption:
            absorption_repr = absorption_repr_many(self.absorption.features)
            feature_repr.append(absorption_repr)
        if self.transits:
            transits_repr = transits_repr_many(self.transits.features)
            feature_repr.append(transits_repr)
        if self.peripherals:
            peripherals_repr = peripherals_repr_many(self.peripherals.features)
            feature_repr.append(peripherals_repr)
        return ';'.join(feature_repr)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
