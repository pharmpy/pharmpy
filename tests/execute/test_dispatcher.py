import pytest

from pharmpy.execute.databases.local_directory import LocalDirectoryDatabase
from pharmpy.execute.dispatcher import ExecutionDispatcher
from pharmpy.execute.dispatchers.local import LocalDispatcher
from pharmpy.tools.workflows import Task, Workflow


def fun(s):
    return s


def test_base_class():
    disp = ExecutionDispatcher()
    with pytest.raises(NotImplementedError):
        disp.run(None, None)


def test_local_dispatcher():
    db = LocalDirectoryDatabase()
    disp = LocalDispatcher()
    wf = Workflow(Task('results', fun, 'input', final_task=True))
    res = disp.run(wf, db)
    assert res == 'input'
