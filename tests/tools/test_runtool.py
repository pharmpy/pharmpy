from typing import Optional

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.model import Model
from pharmpy.tools.run import import_tool
from pharmpy.workflows import Results, Task, Workflow, WorkflowBuilder


def test_import_tool():
    from pharmpy.tools import iivsearch

    tool = import_tool('iivsearch')
    assert tool == iivsearch


def create_workflow_rename(new_name, mock_name=None, model: Optional[Model] = None):
    def rename(m):
        m = m.replace(name=new_name)
        return m

    wb = WorkflowBuilder(tasks=[Task('copy', lambda x: x, model)], name=mock_name)
    wb.insert_workflow(WorkflowBuilder(tasks=[Task('rename', rename)]))
    return Workflow(wb)


@with_same_arguments_as(create_workflow_rename)
def validate_input_rename(model, new_name):
    assert isinstance(new_name, str)
    assert isinstance(model, Model)


def create_workflow_generic(name=None, model: Optional[Model] = None, mock_name=None):
    return Workflow(
        WorkflowBuilder(tasks=[Task('copy', lambda _: Results(), model)], name=mock_name)
    )


@with_same_arguments_as(create_workflow_generic)
def validate_input_generic(model):
    assert isinstance(model, Model)
