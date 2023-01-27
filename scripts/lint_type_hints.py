from inspect import isfunction, getmembers, signature
from typing import get_type_hints

import pharmpy.modeling
import pharmpy.tools


class TypeHintError(Exception):
    pass


funcs = getmembers(pharmpy.modeling, isfunction) + getmembers(pharmpy.tools, isfunction)

for name, func in funcs:
    type_hints = get_type_hints(func)

    # Can be removed if all return types should be annotated
    if 'return' in type_hints.keys():
        del type_hints['return']

    params = signature(func).parameters.values()
    if not type_hints and len(params) > 0:
        raise TypeHintError(f'Type hints missing: {name}')

    # Exclude *args and **kwargs
    param_names = {param.name for param in params if param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)}
    diff = param_names.difference(type_hints.keys())
    if diff:
        raise TypeHintError(f'Not all args have type hints: {name} (args: {diff})')
