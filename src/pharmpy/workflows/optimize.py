def optimize_task_graph_for_dask_distributed(client, graph):
    from dask.distributed import Future

    optimized = {key: _scatter_computation(Future, client, value) for key, value in graph.items()}
    from dask.optimization import fuse

    return fuse(optimized)[0]


def _scatter_computation(Future, client, computation):
    # NOTE According to dask's graph spec (https://docs.dask.org/en/stable/spec.html):
    # A computation may be one of the following:
    #  - Any key present in the Dask graph like 'x'
    #  - Any other value like 1, to be interpreted literally
    #  - A task like (inc, 'x') (see below)
    #  - A list of computations, like [1, 'x', (inc, 'x')]

    if isinstance(computation, tuple):
        return (
            computation[0],
            *map(lambda c: _scatter_computation(Future, client, c), computation[1:]),
        )

    if isinstance(computation, list):
        return list(map(lambda c: _scatter_computation(Future, client, c), computation))

    return _scatter_value(Future, client, computation)


def _scatter_value(Future, client, value):
    # TODO We could automatically compute whether object size is above
    # threshold with a slight twist on https://stackoverflow.com/a/30316760
    if isinstance(value, (int, str, float, bool, range, Future)) or callable(value):
        return value
    else:
        return client.scatter(value)
