from __future__ import annotations

from typing import Iterable, Sequence, Union

from pharmpy.internals.immutable import Immutable

from .features import (
    Absorption,
    Covariate,
    Elimination,
    LagTime,
    ModelFeature,
    Peripherals,
    Transits,
)


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

        grouped_features = {
            Absorption: [],
            Transits: [],
            LagTime: [],
            Elimination: [],
            Peripherals: [],
            Covariate: [],
        }
        for feature in features:
            group = type(feature)
            if group not in grouped_features.keys():
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
    def lagtime(self):
        features = self._filter_by_type(LagTime)
        return ModelFeatures.create(features)

    @property
    def elimination(self):
        features = self._filter_by_type(Elimination)
        return ModelFeatures.create(features)

    @property
    def peripherals(self):
        features = self._filter_by_type(Peripherals)
        return ModelFeatures.create(features)

    @property
    def covariates(self):
        features = self._filter_by_type(Covariate)
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
        # ModelFeatures.create has a canonical order
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
            absorption_repr = Absorption.repr_many(self.absorption.features)
            feature_repr.append(absorption_repr)
        if self.transits:
            transits_repr = Transits.repr_many(self.transits.features)
            feature_repr.append(transits_repr)
        if self.lagtime:
            lagtime_repr = LagTime.repr_many(self.lagtime.features)
            feature_repr.append(lagtime_repr)
        if self.elimination:
            elimination_repr = Elimination.repr_many(self.elimination.features)
            feature_repr.append(elimination_repr)
        if self.peripherals:
            peripherals_repr = Peripherals.repr_many(self.peripherals.features)
            feature_repr.append(peripherals_repr)
        if self.covariates:
            covariates_repr = Covariate.repr_many(self.covariates.features)
            feature_repr.append(covariates_repr)
        return ';'.join(feature_repr)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
