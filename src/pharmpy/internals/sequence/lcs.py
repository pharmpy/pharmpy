from typing import Iterator, Literal, Sequence, Tuple, TypeVar

T = TypeVar('T')
C = Literal[-1, 0, 1]


def diff(old: Sequence[T], new: Sequence[T]) -> Iterator[Tuple[C, T]]:
    """Get diff between a and b in order for all elements

    Optimizes by first handling equal elements from the head and tail
    Each entry is a pair of operation (+1, -1 or 0) and the element
    """
    i = 0
    for a, b in zip(old, new):
        if a == b:
            yield (0, b)
            i += 1
        else:
            break

    rold = old[i:]
    rnew = new[i:]

    saved = []
    for a, b in zip(reversed(rold), reversed(rnew)):
        if a == b:
            saved.append((0, b))
        else:
            break

    rold = rold[: len(rold) - len(saved)]
    rnew = rnew[: len(rnew) - len(saved)]

    c = _matrix(rold, rnew)
    for op, val in _diff(c, rold, rnew, len(rold) - 1, len(rnew) - 1):
        yield op, val

    while saved:
        yield saved.pop()


def _matrix(a: Sequence[T], b: Sequence[T]) -> Sequence[Sequence[int]]:
    # generate matrix of length of longest common subsequence for sublists of both lists
    lengths = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return lengths


def _diff(
    c: Sequence[Sequence[int]], x: Sequence[T], y: Sequence[T], i: int, j: int
) -> Iterator[Tuple[C, T]]:
    """Print the diff using LCS length matrix using backtracking"""
    if i < 0 and j < 0:
        return
    elif i < 0:
        yield from _diff(c, x, y, i, j - 1)
        yield 1, y[j]
    elif j < 0:
        yield from _diff(c, x, y, i - 1, j)
        yield -1, x[i]
    elif x[i] == y[j]:
        yield from _diff(c, x, y, i - 1, j - 1)
        yield 0, x[i]
    elif c[i + 1][j] >= c[i][j + 1]:
        yield from _diff(c, x, y, i, j - 1)
        yield 1, y[j]
    else:
        yield from _diff(c, x, y, i - 1, j)
        yield -1, x[i]
