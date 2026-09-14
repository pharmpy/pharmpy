from collections.abc import Callable, Hashable

from pharmpy.model import Model

FeatureKey = tuple[Hashable, ...]
FeatureFn = Callable[[Model], Model | None]
Feature = tuple[FeatureKey, FeatureFn]
