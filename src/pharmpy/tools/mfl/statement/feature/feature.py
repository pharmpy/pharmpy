from typing import Any, Callable


class ModelFeature:
    pass


def feature(feature_cls: Callable[..., ModelFeature], children: list[Any]) -> ModelFeature:
    return feature_cls(*map(lambda x: tuple(x) if isinstance(x, list) else x, children))
