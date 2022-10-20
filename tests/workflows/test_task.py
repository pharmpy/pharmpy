from pharmpy.workflows import Task


def test_create_tasks():
    t1 = Task('t1', 'func', 'input')
    assert t1.task_input == ('input',)
    t2 = Task('t2', 'func', 1)
    assert t2.task_input == (1,)
    t3 = Task('t3', 'func', 1, 2, 3)
    assert t3.task_input == (1, 2, 3)
    t4 = Task('t4', 'func', [1, 2, 3])
    assert t4.task_input == ([1, 2, 3],)
    t5 = Task('t5', 'func')
    assert t5.task_input == ()
