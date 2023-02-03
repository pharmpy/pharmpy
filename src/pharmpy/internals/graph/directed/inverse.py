from typing import Dict, Set, TypeVar

T = TypeVar('T')
U = TypeVar('U')


def inverse(g: Dict[T, Set[U]]) -> Dict[U, Set[T]]:
    h = {}

    for left, deps in g.items():
        for right in deps:
            if right in h:
                h[right].add(left)
            else:
                h[right] = {left}

    return h
