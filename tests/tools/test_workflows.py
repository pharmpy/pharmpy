import pytest

from pharmpy.tools.workflows import Task, Workflow


@pytest.fixture
def tasks():
    t1 = Task('t1', 'func', 'input')
    t2 = Task('t2', 'func', t1)
    t3 = Task('t3', 'func', t1)
    t4 = Task('t4', 'func', [t2, t3])
    return t1, t2, t3, t4


def test_create_tasks():
    t1 = Task('t1', 'func', 'input')
    assert t1.task_input[0] == 'input'
    t2 = Task('t2', 'func', 1)
    assert t2.task_input[0] == 1
    t3 = Task('t3', 'func', 1, 2, 3)
    assert t3.task_input[2] == 3
    t4 = Task('t4', 'func', [1, 2, 3])
    assert t4.task_input[0] == [1, 2, 3]
    t5 = Task('t5', 'func')
    assert not t5.has_input()


def test_add_tasks(tasks):
    wf = Workflow()

    t1, t2, t3, _ = tasks

    wf.add_tasks(t1)
    assert len(list(wf.tasks.nodes)) == 1

    wf.add_tasks([t2, t3])
    assert len(list(wf.tasks.nodes)) == 3


def test_add_tasks_workflow(tasks):
    t1, t2, t3, _ = tasks

    wf_sequential = Workflow([t1, t2, t3])
    wf_sequential.connect_tasks({t1: [t2, t3]})
    assert len(wf_sequential.tasks.edges) == 2
    assert list(wf_sequential.tasks.successors(t1)) == [t2, t3]

    t4 = Task('t4', 'func', 'input')
    wf_t4 = Workflow(t4)

    wf_sequential.add_tasks(wf_t4, connect=True)
    assert list(wf_sequential.tasks.predecessors(t4)) == [t2, t3]
    assert len(wf_sequential.tasks.edges) == 4
    assert list(wf_sequential.tasks.nodes) == [t1, t2, t3, t4]

    wf_parallel = Workflow([t1, t2, t3])
    wf_parallel.connect_tasks({t1: [t2, t3]})
    wf_parallel.add_tasks(wf_t4, connect=False)
    assert not list(wf_parallel.tasks.predecessors(t4))
    assert len(wf_parallel.tasks.edges) == 2
    assert list(wf_parallel.tasks.nodes) == [t1, t2, t3, t4]

    wf_specified_output_nodes = Workflow([t1, t2, t3])
    wf_specified_output_nodes.connect_tasks({t1: [t2, t3]})
    wf_specified_output_nodes.add_tasks(wf_t4, connect=True, output_nodes=[t3])
    assert list(wf_specified_output_nodes.tasks.predecessors(t4)) == [t3]

    wf_empty_task_input = Workflow([t1, t2, t3])
    wf_empty_task_input.connect_tasks({t1: [t2, t3]})
    t5 = Task('t5', 'func')
    wf_t5 = Workflow(t5)
    assert not t5.has_input()
    wf_empty_task_input.add_tasks(wf_t5, connect=True)
    assert list(wf_empty_task_input.tasks.predecessors(t5)) == [t2, t3]
    assert len(wf_empty_task_input.tasks.edges) == 4
    assert list(wf_empty_task_input.tasks.nodes) == [t1, t2, t3, t5]
    assert t5.task_input[0] == [t2, t3]

    wf_existing_task_input = Workflow([t1, t2, t3])
    wf_existing_task_input.connect_tasks({t1: [t2, t3]})
    t6 = Task('t6', 'func', 'x')
    wf_t6 = Workflow(t6)
    assert t6.has_input()
    wf_existing_task_input.add_tasks(wf_t6, connect=True, arg_index=0)
    assert t6.task_input[0] == [t2, t3]
    assert t6.task_input[1] == 'x'


def test_connect_tasks(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks

    wf.add_tasks([t1, t2, t3, t4], connect=False)
    wf.connect_tasks({t1: [t2, t3], t2: t4, t3: t4})

    assert list(wf.tasks.successors(t1)) == [t2, t3]
    assert list(wf.tasks.predecessors(t4)) == [t2, t3]

    with pytest.raises(ValueError):
        wf.connect_tasks({t1: t1})


def test_get_upstream_tasks(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks
    t5 = Task('t5', 'func', t1)
    t6 = Task('t6', 'func', t3)

    wf.add_tasks(t1)
    wf.add_tasks([t2, t3], connect=True)
    wf.add_tasks(t4, connect=True)
    wf.add_tasks(t5, connect=False)
    wf.connect_tasks({t1: t5, t3: t6})

    assert set(wf.get_upstream_tasks(t4)) == {t1, t2, t3}
    assert set(wf.get_upstream_tasks(t6)) == {t1, t3}
    assert wf.get_upstream_tasks(t5) == [t1]


def test_get_output(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks

    wf.add_tasks([t1, t2, t3])
    wf.connect_tasks({t1: [t2, t3]})

    assert wf.get_output() == [t2, t3]

    wf.add_tasks(t4)
    wf.connect_tasks({t2: t4, t3: t4})

    assert wf.get_output() == [t4]


def test_as_dict(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks

    wf.add_tasks([t1, t2, t3, t4])
    wf.connect_tasks({t1: [t2, t3], t2: t4, t3: t4})

    wf_dict = wf.as_dict()
    wf_keys = list(wf_dict.keys())
    wf_inputs = [task_input for (_, task_input) in wf_dict.values()]

    assert wf_keys[0].startswith('t1') and wf_keys[1].startswith('t2')
    assert wf_inputs[0].startswith('input')
    assert wf_inputs[1].startswith('t1')
    assert isinstance(wf_inputs[3], list)


def test_force_new_task_ids(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks
    wf.add_tasks([t1, t2, t3, t4])
    t1_id, t2_id, t3_id, t4_id = t1.task_id, t2.task_id, t3.task_id, t4.task_id
    wf.force_new_task_ids()

    assert (
        t1_id != t1.task_id and t2_id != t2.task_id and t3_id != t3.task_id and t4_id != t4.task_id
    )


def test_copy(tasks):
    wf = Workflow()

    t1, t2, t3, t4 = tasks

    wf.add_tasks([t1, t2, t3, t4])
    wf.connect_tasks({t1: [t2, t3]})
    wf.connect_tasks({t2: t4, t3: t4})

    task_ids = [task.task_id for task in wf.tasks.nodes]

    wf_copy = wf.copy(new_ids=False)
    assert task_ids == [task.task_id for task in wf_copy.tasks.nodes]

    wf_copy_new_ids = wf.copy(new_ids=True)
    assert task_ids != [task.task_id for task in wf_copy_new_ids.tasks.nodes]
