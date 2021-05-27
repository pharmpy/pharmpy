from pharmpy.tools.workflows import Task, Workflow


def test_as_dict():
    wf = Workflow([Task('t1', 'func', 'x'), Task('t2', 'func', ('t1', 'y'))])

    wf_dict = wf.as_dict()
    wf_keys = list(wf_dict.keys())
    wf_values = list(wf_dict.values())

    assert wf_keys[0].startswith('t1') and wf_keys[1].startswith('t2')
    assert wf_values == [('func', 'x'), ('func', 't1', 'y')]


def test_merge(pheno_path):
    wf_1 = Workflow(Task('t1', 'func', 'x'))
    wf_2 = Workflow(Task('t2', 'func', ('t1', 'y')))

    wf_1.merge_workflows(wf_2)

    wf_dict = wf_1.as_dict()
    wf_keys = list(wf_dict.keys())
    wf_values = list(wf_dict.values())

    assert wf_keys[0].startswith('t1') and wf_keys[1].startswith('t2')
    assert wf_values == [('func', 'x'), ('func', 't1', 'y')]


def test_task():
    t1 = Task('t1', 'func', 'input')
    assert t1.task_input == ['input']
    t2 = Task('t2', 'func', 1)
    assert t2.task_input == [1]
    t3 = Task('t3', 'func', (1, 2, 3))
    assert t3.task_input == [1, 2, 3]
