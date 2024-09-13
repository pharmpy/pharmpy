from typing import Callable, Hashable, Optional

from pharmpy.model import Model

FeatureKey = tuple[Hashable, ...]
FeatureFn = Callable[[Model], Optional[Model]]
Feature = tuple[FeatureKey, FeatureFn]
