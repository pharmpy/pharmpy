import re

import pytest

from pharmpy.workflows import Task, Workflow


@pytest.fixture
def tasks():
    t1 = Task('t1', 'func', 'input')
    t2 = Task('t2', 'func', t1)
    t3 = Task('t3', 'func', t1)
    t4 = Task('t4', 'func', [t2, t3])
    return t1, t2, t3, t4


def test_add_tasks(tasks):
    wf = Workflow()

    t1, t2, t3, _ = tasks

    wf.add_task(t1)
    assert len(wf) == 1

    wf.add_task(t2)
    wf.add_task(t3)
    assert len(wf) == 3


def test_add_tasks_workflow(tasks):
    t1, t2, t3, _ = tasks

    wf_sequential = Workflow([t1])
    wf_sequential.add_task(t2, predecessors=t1)
    wf_sequential.add_task(t3, predecessors=t1)
    assert list(wf_sequential.get_successors(t1)) == [t2, t3]

    t4 = Task('t4', 'func', 'input')
    wf_t4 = Workflow([t4])

    wf_sequential.insert_workflow(wf_t4)
    assert list(wf_sequential.get_predecessors(t4)) == [t2, t3]
    assert len(wf_sequential) == 4

    wf_parallel = Workflow([t1])
    wf_parallel.add_task(t2, predecessors=t1)
    wf_parallel.add_task(t3, predecessors=t1)
    wf_parallel.insert_workflow(wf_t4, predecessors=[])
    assert not list(wf_parallel.get_predecessors(t4))
    assert len(wf_parallel) == 4


def test_get_output(tasks):
    t1, t2, t3, t4 = tasks
    wf = Workflow([t1])
    wf.add_task(t2, predecessors=t1)
    wf.add_task(t3, predecessors=t1)

    assert wf.output_tasks == [t2, t3]

    wf.add_task(t4, predecessors=[t2, t3])

    assert wf.output_tasks == [t4]


def test_as_dask_dict(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks

    wf.add_task(t1)
    wf.add_task(t2, predecessors=t1)
    wf.add_task(t3, predecessors=t1)
    wf.add_task(t4, predecessors=[t2, t3])

    wf_dict = wf.as_dask_dict()
    assert re.match(
        r"\{'.*': \('func', 'input'\), '.*': \('func', t1, '.*'\), '.*': \('func', t1, '.*'\), 'results': \('func', \[t2, t3\], '.*', '.*'\)\}",  # noqa
        str(wf_dict),
    )


def test_as_dask_dict_raises(tasks):
    wf = Workflow(tasks)

    with pytest.raises(ValueError, match='.*one output task.*'):
        wf.as_dask_dict()


def test_insert_workflow(tasks):
    t1, t2, t3, t4 = tasks

    wf = Workflow([t1])
    wf.insert_workflow(Workflow([t2, t3]))
    wf.insert_workflow(Workflow([t4]))

    assert set(wf.input_tasks) == {t1}
    assert set(wf.output_tasks) == {t4}


def test_insert_workflow_with_single_predecessors(tasks):
    t1, t2, t3, t4 = tasks

    wf = Workflow([t1])
    wf.insert_workflow(Workflow([t2, t3]), predecessors=t1)
    wf.insert_workflow(Workflow([t4]), predecessors=t2)

    assert set(wf.input_tasks) == {t1}
    assert set(wf.output_tasks) == {t3, t4}


def test_insert_workflow_with_predecessors(tasks):
    t1, t2, t3, t4 = tasks

    wf = Workflow([t1])
    wf.insert_workflow(Workflow([t2, t3]), predecessors=[t1])
    wf.insert_workflow(Workflow([t4]), predecessors=[t2, t3])

    assert set(wf.input_tasks) == {t1}
    assert set(wf.output_tasks) == {t4}


def test_insert_workflow_nm():
    t1, t2, t3, t4, t5 = map(lambda i: Task(f't{i}', lambda: 0), range(5))
    wf1 = Workflow([t1, t2])
    wf2 = Workflow([t3, t4, t5])
    with pytest.raises(ValueError, match='.*not supported.*'):
        wf1.insert_workflow(wf2)


def test_get_upstream_tasks():
    t1, t2, t3, t4, t5 = map(lambda i: Task(f't{i}', lambda: 0), range(5))
    wf = Workflow([t1, t2])
    wf.insert_workflow(Workflow([t3]))
    wf.insert_workflow(Workflow([t4, t5]))

    assert set(wf.get_upstream_tasks(t4)) == set(wf.get_upstream_tasks(t5)) == {t1, t2, t3}
    assert set(wf.get_upstream_tasks(t1)) == set(wf.get_upstream_tasks(t2)) == set()
    assert set(wf.get_upstream_tasks(t3)) == {t1, t2}


def test_copy(tasks):
    t1, t2, t3, t4 = tasks
    wf = Workflow([t1, t2, t3, t4])
    wf2 = wf.copy()
    assert len(wf) == len(wf2)
    assert str(wf) == str(wf2)


def test_add(tasks):
    t1, t2, t3, t4 = tasks
    wf1 = Workflow([t1])
    wf1.add_task(t2, predecessors=t1)
    wf2 = Workflow([t3])
    wf2.add_task(t4, predecessors=t3)

    wf = wf1 + wf2

    assert len(wf.tasks) == 4
    assert len(wf.input_tasks) == 2
    assert len(wf.output_tasks) == 2
