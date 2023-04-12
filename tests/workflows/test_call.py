import warnings
from uuid import uuid4

import pytest

import pharmpy.workflows.dispatchers
from pharmpy.config import ConfigurationContext
from pharmpy.internals.fs.cwd import chdir
from pharmpy.workflows import Task, Workflow, call_workflow, execute_workflow


def ignore_scratch_warning():
    warnings.filterwarnings(
        "ignore",
        message=".*creating scratch directories is taking a surprisingly long time",
        category=UserWarning,
    )


def sub(a, b):
    t1 = Task('t1', lambda: a)
    t2 = Task('t2', lambda: b)
    t3 = Task('t3', lambda x, y: x + y)
    wf = Workflow([t1, t2])
    wf.add_task(t3, predecessors=[t1, t2])
    return wf


@pytest.mark.xdist_group(name="workflow")
def f(context, a, b):
    wf = sub(a, b)
    name = uuid4()
    with warnings.catch_warnings():
        ignore_scratch_warning()
        res = call_workflow(wf, name, context)
    return res


def add(a, b):
    t1 = Task('t1', lambda: a)
    t2 = Task('t2', lambda: b)
    t3 = Task('t3', f)
    wf = Workflow([t1, t2], name='add')
    wf.add_task(t3, predecessors=[t1, t2])
    return wf


@pytest.mark.xdist_group(name="workflow")
def test_call_workflow_threaded(tmp_path):
    a, b = 1, 2
    wf = add(a, b)

    with ConfigurationContext(pharmpy.workflows.dispatchers.conf, dask_dispatcher='threaded'):
        with chdir(tmp_path):
            with pytest.raises(ValueError, match='No global client found and no address provided'):
                with warnings.catch_warnings():
                    ignore_scratch_warning()
                    execute_workflow(wf)


@pytest.mark.xdist_group(name="workflow")
def test_call_workflow_distributed(tmp_path):
    a, b = 1, 2
    wf = add(a, b)

    with ConfigurationContext(pharmpy.workflows.dispatchers.conf, dask_dispatcher='distributed'):
        with chdir(tmp_path):
            with warnings.catch_warnings():
                ignore_scratch_warning()
                res = execute_workflow(wf)

    assert res == a + b
