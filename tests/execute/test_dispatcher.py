import pytest

from pharmpy.workflows import LocalDirectoryDatabase, Task, Workflow
from pharmpy.workflows.dispatcher import ExecutionDispatcher
from pharmpy.workflows.dispatchers.local import LocalDispatcher


def fun(s):
    return s


def test_base_class():
    disp = ExecutionDispatcher()
    with pytest.raises(NotImplementedError):
        disp.run(None, None)


def test_local_dispatcher():
    db = LocalDirectoryDatabase()
    disp = LocalDispatcher()
    wf = Workflow([Task('results', fun, 'input')])
    res = disp.run(wf, db)
    assert res == 'input'
