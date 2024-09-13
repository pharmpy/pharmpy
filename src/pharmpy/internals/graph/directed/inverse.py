from typing import TypeVar

T = TypeVar('T')
U = TypeVar('U')


def inverse(g: dict[T, set[U]]) -> dict[U, set[T]]:
    h = {}

    for left, deps in g.items():
        for right in deps:
            if right in h:
                h[right].add(left)
            else:
                h[right] = {left}

    return h
