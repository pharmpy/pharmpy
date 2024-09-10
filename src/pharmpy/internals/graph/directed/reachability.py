from typing import Callable, Iterable, TypeVar

T = TypeVar('T')


def reachable_from(start_nodes: set[T], neighbors: Callable[[T], Iterable[T]]) -> set[T]:
    queue = list(start_nodes)
    closure = set(start_nodes)
    while queue:
        u = queue.pop()
        n = neighbors(u)
        for v in n:
            if v not in closure:
                queue.append(v)
                closure.add(v)

    return closure
