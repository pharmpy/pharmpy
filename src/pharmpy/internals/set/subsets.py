from itertools import chain, combinations
from typing import Iterable, Tuple, TypeVar

T = TypeVar('T')


def subsets(iterable: Iterable[T], min_size: int = 0, max_size: int = -1) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the subsets of the input iterable with
    minimum and maximum size constraints. Allows maximum_size to be given
    relatively to iterable "length" by specifying a negative value.

    Adapted from powerset function defined in
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    subsets([1,2,3], min_size=1, max_size=2) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    """
    s = list(iterable)
    max_size = len(s) + max_size + 1 if max_size < 0 else max_size
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size + 1))


def non_empty_proper_subsets(iterable: Iterable[T]) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the non-empty proper subsets of the input
    iterable.

    non_empty_proper_subsets([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
    """
    return subsets(iterable, min_size=1, max_size=-2)


def non_empty_subsets(iterable: Iterable[T]) -> Iterable[Tuple[T]]:
    """Returns an iterable over all the non-empty subsets of the input
    iterable.

    non_empty_subsets([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    """
    return subsets(iterable, min_size=1, max_size=-1)
