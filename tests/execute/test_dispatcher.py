from pharmpy.workflows import Task, Workflow, local_dask


def fun(s):
    return s


def test_local_dispatcher():
    wf = Workflow([Task('results', fun, 'input')])
    res = local_dask.run(wf)
    assert res == 'input'
