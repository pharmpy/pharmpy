from typing import Callable, Hashable, Optional, Tuple

from pharmpy.model import Model

FeatureKey = Tuple[Hashable, ...]
FeatureFn = Callable[[Model], Optional[Model]]
Feature = Tuple[FeatureKey, FeatureFn]
