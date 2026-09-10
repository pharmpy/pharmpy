from collections.abc import Callable, Iterable

from .reachability import reachable_from


def strongly_connected_component_of[T](
    vertex: T, successors: Callable[[T], Iterable[T]], predecessors: Callable[[T], Iterable[T]]
):
    forward_reachable = reachable_from({vertex}, successors)

    # NOTE: This searches for backward reachable vertices on the graph induced
    # by the forward reachable vertices and is equivalent to (but less wasteful
    # than) first computing the backward reachable vertices on the original
    # graph and then computing the intersection with the forward reachable
    # vertices.
    return reachable_from(
        {vertex},
        lambda u: filter(forward_reachable.__contains__, predecessors(u)),
    )
