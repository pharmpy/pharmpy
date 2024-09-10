from typing import Dict, TypeVar

T = TypeVar('T')
U = TypeVar('U')


def inverse(g: Dict[T, set[U]]) -> Dict[U, set[T]]:
    h = {}

    for left, deps in g.items():
        for right in deps:
            if right in h:
                h[right].add(left)
            else:
                h[right] = {left}

    return h
