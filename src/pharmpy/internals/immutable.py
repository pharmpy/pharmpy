from abc import ABC
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple, TypeVar, Union, cast


class Immutable(ABC):
    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self


K = TypeVar('K')
V = TypeVar('V')


def frozenmapping(*args: Union[Mapping[K, V], Iterable[Tuple[K, V]]], **kwargs: V) -> Mapping[K, V]:
    """Returns a frozen 'dict'. Missing a hash function at the moment."""

    if len(args) >= 2:
        # NOTE We make sure any call to dict does not raise later
        raise TypeError(f'frozenmapping expected at most 1 argument, got {len(args)}')

    arg = args[0] if args else ()

    if kwargs or not isinstance(arg, MappingProxyType):
        return MappingProxyType(dict(arg, **kwargs))

    # NOTE MappingProxyType[K, V] is both a Mapping[K, V] and an Iterable[K].
    # Pyright tries to idenfiy Iterable[Tuple[K, V]] to an input of type
    # MappingProxyType[Tuple[K, V], Any] which is unfortunately not what we
    # want. We also cannot cast to MappingProxyType[K, V] because Python 3.8
    # does not define that.
    return cast(MappingProxyType, arg)
