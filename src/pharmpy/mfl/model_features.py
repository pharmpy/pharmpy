from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Iterable, Sequence, Union

from lark import UnexpectedInput

from pharmpy.internals.immutable import Immutable
from pharmpy.mfl.features.mutex_feature import MutexFeature

from .features import (
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


class ModelFeatures(Immutable):
    def __init__(self, features: tuple[ModelFeature, ...]):
        self._features = features

    @classmethod
    def create(cls, features: Iterable[ModelFeature] | str):
        if isinstance(features, str):
            try:
                features = parse(features)
            except UnexpectedInput:
                raise ValueError(f'Could not parse string `features`: got {features}')
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
            Allometry: [],
            DirectEffect: [],
            IndirectEffect: [],
            EffectComp: [],
            Metabolite: [],
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
    def pk_iv(cls):
        transits = [Transits.create(0)]
        lagtime = [LagTime.create(on=False)]
        elimination = [Elimination.create(type='FO')]
        peripherals = [Peripherals.create(n) for n in range(0, 3)]
        features = transits + lagtime + elimination + peripherals
        return cls.create(features=features)

    @classmethod
    def pk_oral(cls):
        absorption = [Absorption.create(type=type) for type in ('FO', 'ZO', 'SEQ-ZO-FO')]
        transits = [
            Transits.create(n, with_depot=depot)
            for n, depot in itertools.product([0, 1, 3, 10], [True, False])
        ]
        lagtime = [LagTime.create(on=False), LagTime.create(on=True)]
        elimination = [Elimination.create(type='FO')]
        peripherals = [Peripherals.create(n) for n in range(0, 2)]
        features = absorption + transits + lagtime + elimination + peripherals
        return cls.create(features=features)

    @classmethod
    def pd(cls):
        types = ['LINEAR', 'EMAX', 'SIGMOID']
        direct_effects = [DirectEffect.create(type=type) for type in types]
        indirect_effects = [
            IndirectEffect.create(type=type, production_type=production_type)
            for type, production_type in itertools.product(types, ['PRODUCTION', 'DEGRADATION'])
        ]
        effect_compartments = [EffectComp.create(type=type) for type in types]
        features = direct_effects + indirect_effects + effect_compartments
        return cls.create(features=features)

    @property
    def features(self):
        return self._features

    @property
    def absorption(self):
        return self._get_feature_type(Absorption)

    @property
    def transits(self):
        return self._get_feature_type(Transits)

    @property
    def lagtime(self):
        return self._get_feature_type(LagTime)

    @property
    def elimination(self):
        return self._get_feature_type(Elimination)

    @property
    def peripherals(self):
        return self._get_feature_type(Peripherals)

    @property
    def covariates(self):
        return self._get_feature_type(Covariate)

    @property
    def allometry(self):
        return self._get_feature_type(Allometry)

    @property
    def direct_effects(self):
        return self._get_feature_type(DirectEffect)

    @property
    def indirect_effects(self):
        return self._get_feature_type(IndirectEffect)

    @property
    def effect_compartments(self):
        return self._get_feature_type(EffectComp)

    @property
    def metabolites(self):
        return self._get_feature_type(Metabolite)

    def _get_feature_type(self, type):
        features = []
        for feature in self.features:
            if isinstance(feature, type):
                features.append(feature)
        return ModelFeatures.create(features)

    def is_expanded(self):
        for feature in self.features:
            arg_types = (type(arg) for arg in feature.args)
            if Ref in arg_types:
                return False
        return True

    def is_single_model(self):
        feature_map = defaultdict(list)
        for feature in self.features:
            features = feature_map[type(feature)]
            if isinstance(feature, (MutexFeature, IndirectEffect, Transits)):
                if len(features) > 0:
                    return False
            elif isinstance(feature, Peripherals):
                if len(features) == 1 and feature.type == features[0].type:
                    return False
                if len(features) > 1:
                    return False
                print(features)
            elif isinstance(feature, Covariate):
                if feature.optional:
                    return False
            else:
                raise NotImplementedError
            feature_map[type(feature)].append(feature)
        return True

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

        return ';'.join(feature_repr)


def _flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result
