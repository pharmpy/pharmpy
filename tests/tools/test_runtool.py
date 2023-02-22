from dataclasses import dataclass
from typing import Callable, Optional

import pytest

from pharmpy.internals.fn.signature import with_same_arguments_as
from pharmpy.internals.fs.cwd import chdir
from pharmpy.model import Model, Results
from pharmpy.tools.run import import_tool, run_tool_with_name
from pharmpy.workflows import Task, Workflow


def test_import_tool():
    from pharmpy.tools import iivsearch

    tool = import_tool('iivsearch')
    assert tool == iivsearch


def create_workflow_rename(new_name, name=None, model: Optional[Model] = None):
    def rename(m):
        m = m.replace(name=new_name)
        return m

    wf = Workflow([Task('copy', lambda x: x, model)], name=name)
    wf.insert_workflow(Workflow([Task('rename', rename)]))
    return wf


@with_same_arguments_as(create_workflow_rename)
def validate_input_rename(model, new_name):
    assert isinstance(new_name, str)
    assert isinstance(model, Model)


def create_workflow_generic(name=None, model: Optional[Model] = None):
    return Workflow([Task('copy', lambda _: Results(), model)], name=name)


@with_same_arguments_as(create_workflow_generic)
def validate_input_generic(model):
    assert isinstance(model, Model)


@dataclass(frozen=True)
class MockedTool:
    create_workflow: Callable[..., Workflow]


@dataclass(frozen=True)
class MockedToolWithInputValidation(MockedTool):
    validate_input: Callable[..., None]


@pytest.mark.parametrize(
    ('name', 'tool', 'args', 'expected'),
    (
        ('mocked', MockedTool(create_workflow_generic), (), lambda res: isinstance(res, Results)),
        (
            'mocked',
            MockedToolWithInputValidation(create_workflow_generic, validate_input_generic),
            (),
            lambda res: isinstance(res, Results),
        ),
        (
            'modelfit',  # NOTE This triggers the modelfit-specific (non-)branches
            MockedTool(create_workflow_rename),
            ('y',),
            lambda res: isinstance(res, Model) and res.name == 'y',
        ),
        (
            'modelfit',  # NOTE This triggers the modelfit-specific (non-)branches
            MockedToolWithInputValidation(create_workflow_rename, validate_input_rename),
            ('y',),
            lambda res: isinstance(res, Model) and res.name == 'y',
        ),
    ),
)
def test_run_tool_without_input_validation(tmp_path, pheno, name, tool, args, expected):
    with chdir(tmp_path):
        res = run_tool_with_name(name, tool, args, {'name': name, 'model': pheno})
        assert expected(res)
