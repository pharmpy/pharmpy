"""
:meta private:
"""

import importlib
import inspect
import re
from functools import wraps
from typing import Callable

from .run import run_tool


def wrap(tool_name: str) -> Callable:
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

    # NOTE override the signature to include **kwargs which are used in run_tool, needed for pharmr
    func.__signature__ = _append_kwargs_to_sig(inspect.signature(create_workflow))

    # NOTE add kwargs to docstring
    assert func.__doc__ is not None
    assert run_tool.__doc__ is not None
    func.__doc__ = _append_kwargs_to_doc(func.__doc__, run_tool.__doc__)

    return func


def _append_kwargs_to_sig(sig: inspect.Signature) -> inspect.Signature:
    param_kwargs = inspect.Parameter('kwargs', inspect.Parameter.VAR_KEYWORD)
    return sig.replace(parameters=tuple(sig.parameters.values()) + (param_kwargs,))


def _append_kwargs_to_doc(doc_wrapper: str, doc_run_tool: str) -> str:
    # NOTE get where in docstring to add kwargs documentation
    m_wrapper = re.compile(r'(.)\s*Returns*\s*\n\s*-------')
    search_wrapper = re.search(m_wrapper, doc_wrapper)
    assert search_wrapper is not None

    # NOTE get documentation for kwargs from run_tool
    m_run_tool = re.compile(r'(\s*kwargs\n(.+\n))\n.\s*Returns*\s*\n\s*-+')
    search_run_tool = re.search(m_run_tool, doc_run_tool)
    assert search_run_tool is not None

    return doc_wrapper.replace(search_wrapper.group(0), search_run_tool.group(0))
