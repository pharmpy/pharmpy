import re
from inspect import isfunction, getmembers, signature
from typing import get_type_hints

import pharmpy.modeling
import pharmpy.tools


class TypeHintError(Exception):
    pass


def extract_parameters_from_docstring(func):
    lines = func.__doc__.splitlines()

    def find_first_param_line(s):
        for i, line in enumerate(s):
            if line.lstrip() == 'Parameters':
                return i + 2
        raise ValueError(f"Could not find Parameters documentation of {func.__name__}")

    def find_last_param_line(s, first):
        for i, line in enumerate(s[first:]):
            if line.lstrip() == "":
                return i + first - 1
        return i + first

    def leading_spaces(s):
        return len(s) - len(s.lstrip())

    first_line = find_first_param_line(lines)
    last_line = find_last_param_line(lines, first_line)
    params = lines[first_line:last_line]
    base_indentation = leading_spaces(params[0])
    args = [re.split(r'[\s:]', line.lstrip(), maxsplit=1)[0] for line in params if leading_spaces(line) == base_indentation]
    return args


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

    if param_names:
        args = extract_parameters_from_docstring(func)
        diff = param_names.difference(args)
        if diff:
            raise ValueError(f'Not all args are documented: {name} (args: {diff})')
