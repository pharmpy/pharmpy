from collections.abc import Callable, Iterable


def reachable_from[T](start_nodes: set[T], neighbors: Callable[[T], Iterable[T]]) -> set[T]:
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
