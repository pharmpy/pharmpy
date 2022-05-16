"""
:meta private:
"""

import importlib
import inspect
import pkgutil
import sys

import pharmpy.tools as tools


def _split_to_args_kwargs(params):
    args = []
    kwargs = {}
    for param in params.values():
        if param.default == param.empty:
            args.append(f'{param.name}')
        else:
            kwargs[f'{param.name}'] = param.default
    return args, kwargs


def _create_function(name, sig):
    from pharmpy.modeling import run_tool

    args, kwargs = _split_to_args_kwargs(sig.parameters)

    def _func(*args, **kwargs):
        return run_tool(name, *args, **kwargs)

    _func.__signature__ = sig
    return _func


for module in pkgutil.iter_modules(tools.__path__):
    if module.ispkg:
        tool_name = module.name
        tool_module = importlib.import_module(f'pharmpy.tools.{tool_name}')

        if tool_name == 'modelfit':
            continue
        if 'create_workflow' not in dir(tool_module):
            continue
        if tool_module.create_workflow.__doc__ is None:
            continue

        create_workflow = tool_module.create_workflow
        tool_sig = inspect.signature(create_workflow)

        func = _create_function(tool_name, tool_sig)

        wrapper_name = f'run_{tool_name}'
        func.__name__ = wrapper_name
        func.__doc__ = create_workflow.__doc__

        current_module = sys.modules[__name__]
        setattr(current_module, wrapper_name, func)
