from pharmpy.workflows import Task, Workflow, local_dask


def test_local_dispatcher():
    wf = Workflow([Task('results', lambda x: x, 'input')])
    res = local_dask.run(wf)
    assert res == 'input'
