from collections.abc import Callable, Hashable
from typing import Optional

from pharmpy.model import Model

FeatureKey = tuple[Hashable, ...]
FeatureFn = Callable[[Model], Optional[Model]]
Feature = tuple[FeatureKey, FeatureFn]
