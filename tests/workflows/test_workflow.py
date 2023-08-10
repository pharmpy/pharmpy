import re

import pytest

from pharmpy.workflows import Task, Workflow, WorkflowBuilder


@pytest.fixture
def tasks():
    t1 = Task('t1', 'func', 'input')
    t2 = Task('t2', 'func', t1)
    t3 = Task('t3', 'func', t1)
    t4 = Task('t4', 'func', [t2, t3])
    return t1, t2, t3, t4


def test_add_tasks(tasks):
    wb = WorkflowBuilder()

    t1, t2, t3, _ = tasks

    wb.add_task(t1)
    assert len(wb) == 1

    wb.add_task(t2)
    wb.add_task(t3)
    assert len(wb) == 3


def test_add_tasks_workflow(tasks):
    t1, t2, t3, _ = tasks

    wb_sequential = WorkflowBuilder(tasks=[t1])
    wb_sequential.add_task(t2, predecessors=t1)
    wb_sequential.add_task(t3, predecessors=t1)
    assert list(Workflow(wb_sequential).get_successors(t1)) == [t2, t3]

    t4 = Task('t4', 'func', 'input')
    wb_t4 = WorkflowBuilder(tasks=[t4])

    wb_sequential.insert_workflow(wb_t4)
    assert list(Workflow(wb_sequential).get_predecessors(t4)) == [t2, t3]
    assert len(wb_sequential) == 4

    wb_parallel = WorkflowBuilder(tasks=[t1])
    wb_parallel.add_task(t2, predecessors=t1)
    wb_parallel.add_task(t3, predecessors=t1)
    wb_parallel.insert_workflow(Workflow(wb_t4), predecessors=[])
    assert not list(Workflow(wb_parallel).get_predecessors(t4))
    assert len(wb_parallel) == 4


def test_get_output(tasks):
    t1, t2, t3, t4 = tasks
    wb = WorkflowBuilder(tasks=[t1])
    wb.add_task(t2, predecessors=t1)
    wb.add_task(t3, predecessors=t1)

    assert Workflow(wb).output_tasks == [t2, t3]

    wb.add_task(t4, predecessors=[t2, t3])

    assert Workflow(wb).output_tasks == [t4]


def test_as_dask_dict(tasks):
    wb = WorkflowBuilder()

    t1, t2, t3, t4 = tasks

    wb.add_task(t1)
    wb.add_task(t2, predecessors=t1)
    wb.add_task(t3, predecessors=t1)
    wb.add_task(t4, predecessors=[t2, t3])

    wf_dict = Workflow(wb).as_dask_dict()
    assert re.match(
        r"\{'.*': \('func', 'input'\), '.*': \('func', t1, '.*'\), '.*': \('func', t1, '.*'\), 'results': \('func', \[t2, t3\], '.*', '.*'\)\}",  # noqa
        str(wf_dict),
    )


def test_as_dask_dict_raises(tasks):
    wb = WorkflowBuilder(tasks=tasks)

    with pytest.raises(ValueError, match='.*one output task.*'):
        Workflow(wb).as_dask_dict()


def test_insert_workflow(tasks):
    t1, t2, t3, t4 = tasks

    wb = WorkflowBuilder(tasks=[t1])
    wb.insert_workflow(WorkflowBuilder(tasks=[t2, t3]))
    wb.insert_workflow(WorkflowBuilder(tasks=[t4]))

    assert set(wb.input_tasks) == {t1}
    assert set(wb.output_tasks) == {t4}


def test_insert_workflow_with_single_predecessors(tasks):
    t1, t2, t3, t4 = tasks

    wb = WorkflowBuilder(tasks=[t1])
    wb.insert_workflow(WorkflowBuilder(tasks=[t2, t3]), predecessors=t1)
    wb.insert_workflow(WorkflowBuilder(tasks=[t4]), predecessors=t2)

    assert set(wb.input_tasks) == {t1}
    assert set(wb.output_tasks) == {t3, t4}


def test_insert_workflow_with_predecessors(tasks):
    t1, t2, t3, t4 = tasks

    wb = WorkflowBuilder(tasks=[t1])
    wb.insert_workflow(WorkflowBuilder(tasks=[t2, t3]), predecessors=[t1])
    wb.insert_workflow(WorkflowBuilder(tasks=[t4]), predecessors=[t2, t3])

    assert set(wb.input_tasks) == {t1}
    assert set(wb.output_tasks) == {t4}


def test_insert_workflow_nm():
    t1, t2, t3, t4, t5 = map(lambda i: Task(f't{i}', lambda: 0), range(5))
    wb1 = WorkflowBuilder(tasks=[t1, t2])
    wb2 = WorkflowBuilder(tasks=[t3, t4, t5])
    with pytest.raises(ValueError, match='.*not supported.*'):
        wb1.insert_workflow(wb2)


def test_get_upstream_tasks():
    t1, t2, t3, t4, t5 = map(lambda i: Task(f't{i}', lambda: 0), range(5))
    wb = WorkflowBuilder(tasks=[t1, t2])
    wb.insert_workflow(WorkflowBuilder(tasks=[t3]))
    wb.insert_workflow(WorkflowBuilder(tasks=[t4, t5]))

    wf = Workflow(wb)
    assert set(wf.get_upstream_tasks(t4)) == set(wf.get_upstream_tasks(t5)) == {t1, t2, t3}
    assert set(wf.get_upstream_tasks(t1)) == set(wf.get_upstream_tasks(t2)) == set()
    assert set(wf.get_upstream_tasks(t3)) == {t1, t2}


def test_add(tasks):
    t1, t2, t3, t4 = tasks
    wb1 = WorkflowBuilder(tasks=[t1])
    wb1.add_task(t2, predecessors=t1)
    wb2 = WorkflowBuilder(tasks=[t3])
    wb2.add_task(t4, predecessors=t3)

    wb = wb1 + wb2

    wf = Workflow(wb)
    assert len(wf.tasks) == 4
    assert len(wf.input_tasks) == 2
    assert len(wf.output_tasks) == 2
