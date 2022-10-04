"""
:meta private:
"""

import importlib
from functools import wraps
from typing import Callable

from .run import run_tool


def wrap(tool_name: str) -> Callable:

    if tool_name == 'modelfit':
        raise ValueError('Cannot wrap `modelfit` tool.')

    tool_module = importlib.import_module(f'pharmpy.tools.{tool_name}')

    if 'create_workflow' not in dir(tool_module):
        raise ValueError(f'Tool `{tool_name}` does not export `create_workflow`.')

    create_workflow = tool_module.create_workflow

    if create_workflow.__doc__ is None:
        raise ValueError(
            f'Tool `{tool_name}` does export `create_workflow`'
            ' but this `create_workflow` is lacking `__doc__`.'
        )

    return _create_wrap(tool_name, create_workflow)


def _create_wrap(tool_name: str, create_workflow: Callable) -> Callable:
    @wraps(create_workflow)
    def func(*args, **kwargs):
        return run_tool(tool_name, *args, **kwargs)

    func.__name__ = f'run_{tool_name}'
    return func
