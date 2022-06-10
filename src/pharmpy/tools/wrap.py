"""
:meta private:
"""

import importlib
import inspect
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
    tool_sig = inspect.signature(create_workflow)

    func = _create_function(tool_name, tool_sig)

    wrapper_name = f'run_{tool_name}'
    func.__name__ = wrapper_name
    func.__doc__ = create_workflow.__doc__
    return func


def _create_function(tool_name: str, sig) -> Callable:
    def _func(*args, **kwargs):
        return run_tool(tool_name, *args, **kwargs)

    _func.__signature__ = sig
    return _func
