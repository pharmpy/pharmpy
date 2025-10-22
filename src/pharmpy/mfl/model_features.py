from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, Iterator, Sequence, Type, TypeVar, Union

from lark import UnexpectedInput

from pharmpy.internals.immutable import Immutable
from pharmpy.mfl.features.mutex_feature import MutexFeature

from .features import (
    IIV,
    IOV,
    Absorption,
    Allometry,
    Covariate,
    DirectEffect,
    EffectComp,
    Elimination,
    IndirectEffect,
    LagTime,
    Metabolite,
    ModelFeature,
    Peripherals,
    Ref,
    Transits,
)
from .parsing import parse

T = TypeVar("T", bound=ModelFeature)


class ModelFeatures(Immutable):
    def __init__(self, features: tuple[ModelFeature, ...]):
        self._features = features

    @classmethod
    def create(cls, features: Iterable[ModelFeature] | str) -> ModelFeatures:
        if isinstance(features, str):
            try:
                features = parse(features)
            except UnexpectedInput:
                raise ValueError(f'Could not parse string `features`: got {features}')
        if not isinstance(features, Iterable):
            raise TypeError(f'Type of `feature` must be an iterable: got {type(features)}')
        other_types = {str(type(f)) for f in features if not isinstance(f, ModelFeature)}
        if other_types:
            raise TypeError(f'Incorrect types in `features`: got {sorted(other_types)}')

        grouped_features = {
            Absorption: [],
            Transits: [],
            LagTime: [],
            Elimination: [],
            Peripherals: [],
            Covariate: [],
            Allometry: [],
            DirectEffect: [],
            IndirectEffect: [],
            EffectComp: [],
            Metabolite: [],
            IIV: [],
            IOV: [],
        }
        for feature in features:
            group = type(feature)
            if group not in grouped_features.keys():
                raise NotImplementedError
            if feature not in grouped_features[group]:
                if group == Allometry and len(grouped_features[group]) > 0:
                    raise ValueError('Invalid `features`: Got more than one Allometry feature')
                grouped_features[group].append(feature)

        features = _flatten([sorted(value) for value in grouped_features.values()])
        return cls(features=tuple(features))

    def replace(self, **kwargs) -> ModelFeatures:
        features = kwargs.get('features', self._features)
        return ModelFeatures.create(features=features)

    @classmethod
    def pk_iv(cls) -> ModelFeatures:
        transits = [Transits.create(0)]
        lagtime = [LagTime.create(on=False)]
        elimination = [Elimination.create(type='FO')]
        peripherals = [Peripherals.create(n) for n in range(0, 3)]
        features = transits + lagtime + elimination + peripherals
        return cls.create(features=features)

    @classmethod
    def pk_oral(cls) -> ModelFeatures:
        absorption = [Absorption.create(type=type) for type in ('FO', 'ZO', 'SEQ-ZO-FO')]
        transits = [
            Transits.create(n, depot=depot)
            for n, depot in itertools.product([0, 1, 3, 10], [True, False])
        ]
        lagtime = [LagTime.create(on=False), LagTime.create(on=True)]
        elimination = [Elimination.create(type='FO')]
        peripherals = [Peripherals.create(n) for n in range(0, 2)]
        features = absorption + transits + lagtime + elimination + peripherals
        return cls.create(features=features)

    @classmethod
    def pd(cls) -> ModelFeatures:
        types = ['LINEAR', 'EMAX', 'SIGMOID']
        direct_effects = [DirectEffect.create(type=type) for type in types]
        indirect_effects = [
            IndirectEffect.create(type=type, production=production)
            for type, production in itertools.product(types, [True, False])
        ]
        effect_compartments = [EffectComp.create(type=type) for type in types]
        features = direct_effects + indirect_effects + effect_compartments
        return cls.create(features=features)

    @property
    def features(self) -> tuple[ModelFeature, ...]:
        return self._features

    @property
    def absorption(self) -> ModelFeatures:
        return self._get_feature_type(Absorption)

    @property
    def transits(self) -> ModelFeatures:
        return self._get_feature_type(Transits)

    @property
    def lagtime(self) -> ModelFeatures:
        return self._get_feature_type(LagTime)

    @property
    def elimination(self) -> ModelFeatures:
        return self._get_feature_type(Elimination)

    @property
    def peripherals(self) -> ModelFeatures:
        return self._get_feature_type(Peripherals)

    @property
    def covariates(self) -> ModelFeatures:
        return self._get_feature_type(Covariate)

    @property
    def allometry(self) -> ModelFeatures:
        return self._get_feature_type(Allometry)

    @property
    def direct_effects(self) -> ModelFeatures:
        return self._get_feature_type(DirectEffect)

    @property
    def indirect_effects(self) -> ModelFeatures:
        return self._get_feature_type(IndirectEffect)

    @property
    def effect_compartments(self) -> ModelFeatures:
        return self._get_feature_type(EffectComp)

    @property
    def metabolites(self) -> ModelFeatures:
        return self._get_feature_type(Metabolite)

    @property
    def iiv(self) -> ModelFeatures:
        return self._get_feature_type(IIV)

    @property
    def iov(self) -> ModelFeatures:
        return self._get_feature_type(IOV)

    def _get_feature_type(self, type: Type[T]) -> ModelFeatures:
        features = []
        for feature in self:
            if isinstance(feature, type):
                features.append(feature)
        return ModelFeatures.create(features)

    @property
    def refs(self) -> tuple[Ref, ...]:
        refs = [arg for feature in self for arg in feature.args if isinstance(arg, Ref)]
        return tuple(sorted(set(refs)))

    def is_expanded(self) -> bool:
        for feature in self:
            if not feature.is_expanded():
                return False
        return True

    def is_single_model(self) -> bool:
        feature_map = defaultdict(list)
        for feature in self:
            features = feature_map[type(feature)]
            if isinstance(feature, (MutexFeature, IndirectEffect, Transits, LagTime)):
                if len(features) > 0:
                    return False
            elif isinstance(feature, Peripherals):
                if len(features) == 1 and feature.metabolite == features[0].metabolite:
                    return False
                if len(features) > 1:
                    return False
            elif isinstance(feature, (Covariate, IIV, IOV)):
                if feature.optional:
                    return False
            else:
                raise NotImplementedError
            feature_map[type(feature)].append(feature)
        return True

    def expand(self, expand_to: dict[Ref, Iterable[str]]) -> ModelFeatures:
        if self.is_expanded():
            return self

        features_new = []
        for feature in self:
            if feature.is_expanded():
                features_new.append(feature)
            else:
                features_new.extend(feature.expand(expand_to))

        return self.create(features=features_new)

    def __add__(
        self, other: Union[ModelFeature, ModelFeatures, Iterable[ModelFeature]]
    ) -> ModelFeatures:
        if isinstance(other, ModelFeature):
            return ModelFeatures.create(features=self.features + (other,))
        elif isinstance(other, ModelFeatures):
            return ModelFeatures.create(features=self.features + other.features)
        elif isinstance(other, Sequence):
            return ModelFeatures.create(features=self.features + tuple(other))
        else:
            return NotImplemented

    def __radd__(
        self, other: Union[ModelFeature, ModelFeatures, Iterable[ModelFeature]]
    ) -> ModelFeatures:
        # ModelFeatures.create has a canonical order
        return self + other

    def __sub__(
        self, other: Union[ModelFeature, ModelFeatures, Iterable[ModelFeature]]
    ) -> ModelFeatures:
        if isinstance(other, ModelFeature):
            return ModelFeatures(features=tuple(feature for feature in self if feature != other))
        elif isinstance(other, (ModelFeatures, Sequence)):
            return ModelFeatures(
                features=tuple(feature for feature in self if feature not in other)
            )
        else:
            return NotImplemented

    def __rsub__(
        self, other: Union[ModelFeature, ModelFeatures, Iterable[ModelFeature]]
    ) -> ModelFeatures:
        if isinstance(other, ModelFeature):
            return ModelFeatures(tuple()) if other in self else ModelFeatures((other,))
        elif isinstance(other, Sequence):
            return ModelFeatures(
                features=tuple(feature for feature in other if feature not in self)
            )
        else:
            return NotImplemented

    def __iter__(self) -> Iterator[ModelFeature]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __contains__(
        self, item: Union[ModelFeature, ModelFeatures, Iterable[ModelFeature]]
    ) -> bool:
        if isinstance(item, ModelFeature):
            if item in self.features:
                return True
        elif isinstance(item, ModelFeatures):
            if all(feature in self.features for feature in item):
                return True
        elif isinstance(item, Iterable):
            if all(isinstance(x, ModelFeature) and x in self.features for x in item):
                return True
        else:
            return False
        return False

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, ModelFeatures):
            return False
        if len(self) != len(other):
            return False
        return all(feat1 == feat2 for feat1, feat2 in zip(self.features, other.features))

    def __repr__(self) -> str:
        feature_repr = []
        if self.absorption:
            absorption_repr = Absorption.repr_many(self.absorption)
            feature_repr.append(absorption_repr)
        if self.transits:
            transits_repr = Transits.repr_many(self.transits)
            feature_repr.append(transits_repr)
        if self.lagtime:
            lagtime_repr = LagTime.repr_many(self.lagtime)
            feature_repr.append(lagtime_repr)
        if self.elimination:
            elimination_repr = Elimination.repr_many(self.elimination)
            feature_repr.append(elimination_repr)
        if self.peripherals:
            peripherals_repr = Peripherals.repr_many(self.peripherals)
            feature_repr.append(peripherals_repr)
        if self.covariates:
            covariates_repr = Covariate.repr_many(self.covariates)
            feature_repr.append(covariates_repr)
        if self.allometry:
            allometry_repr = Allometry.repr_many(self.allometry)
            feature_repr.append(allometry_repr)
        if self.direct_effects:
            direct_effect_repr = DirectEffect.repr_many(self.direct_effects)
            feature_repr.append(direct_effect_repr)
        if self.indirect_effects:
            indirect_effect_repr = IndirectEffect.repr_many(self.indirect_effects)
            feature_repr.append(indirect_effect_repr)
        if self.effect_compartments:
            effect_comp_repr = EffectComp.repr_many(self.effect_compartments)
            feature_repr.append(effect_comp_repr)
        if self.metabolites:
            metabolite_repr = Metabolite.repr_many(self.metabolites)
            feature_repr.append(metabolite_repr)
        if self.iiv:
            iiv_repr = IIV.repr_many(self.iiv)
            feature_repr.append(iiv_repr)
        if self.iov:
            iov_repr = IOV.repr_many(self.iov)
            feature_repr.append(iov_repr)

        return ';'.join(feature_repr)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
