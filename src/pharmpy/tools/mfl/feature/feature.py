from typing import Callable, Hashable, Tuple

from pharmpy.model import Model

FeatureKey = Tuple[Hashable, ...]
FeatureFn = Callable[[Model], None]
Feature = Tuple[FeatureKey, FeatureFn]
