from collections.abc import Callable
from typing import Any


class ModelFeature:
    pass


def feature(feature_cls: Callable[..., ModelFeature], children: list[Any]) -> ModelFeature:
    return feature_cls(*(tuple(x) if isinstance(x, list) else x for x in children))
