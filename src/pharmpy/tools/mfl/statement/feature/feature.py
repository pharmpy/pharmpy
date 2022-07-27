from typing import Any, Callable, List


class ModelFeature:
    pass


def feature(feature_cls: Callable[..., ModelFeature], children: List[Any]) -> ModelFeature:
    return feature_cls(*map(lambda x: tuple(x) if isinstance(x, list) else x, children))
