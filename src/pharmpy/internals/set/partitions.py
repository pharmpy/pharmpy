from typing import Iterable, Iterator, List, Sequence, Sized, Tuple, TypeVar

T = TypeVar('T')
S = TypeVar('S', bound=Sized)


def partitions(elements: Iterable[T]) -> Iterator[Tuple[Tuple[T, ...], ...]]:
    """Returns all partitions of a set of elements

    Each partition is represented canonically as a shortlex-sorted tuple of
    tuples. Partitions are ordered by length, then length of parts, and finally
    lexicographically.
    """
    _elements = tuple(elements)
    n = len(_elements)
    return iter(
        sorted(
            map(
                tuple,
                map(
                    _shortlexsorted,
                    _partitions(_elements, n),  # pyright: ignore [reportGeneralTypeIssues]
                ),
            ),
            key=_partitionkey,
        )
    )


def _partitions(elements: Sequence[T], n: int) -> Iterator[Tuple[Tuple[T, ...], ...]]:
    if n == 0:
        yield ()

    else:
        last = elements[n - 1]
        suffix = (last,)
        for partition in _partitions(elements, n - 1):
            yield partition + (suffix,)
            for i, part in enumerate(partition):
                yield partition[:i] + (part + suffix,) + partition[i + 1 :]


def _partitionkey(x: Tuple[S]) -> Tuple[int, Tuple[int, ...], Tuple[S]]:
    return (len(x), tuple(map(len, x)), x)


def _shortlexkey(x: S) -> Tuple[int, S]:
    return (len(x), x)


def _shortlexsorted(iterable: Iterable[S]) -> List[S]:
    return sorted(iterable, key=_shortlexkey)
