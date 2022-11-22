from __future__ import annotations

import re
from dataclasses import dataclass

_re_underscore = re.compile(r'_')


def nonmem_to_pharmpy(name: str):
    parsed_name = _parse_nonmem_variable_name(name)

    if isinstance(parsed_name, ArrayItemName):
        return f'{parsed_name.array}_{parsed_name.index}'

    if isinstance(parsed_name, MatrixItemName):
        return f'{parsed_name.matrix}_{parsed_name.i}_{parsed_name.j}'

    assert isinstance(parsed_name, VariableName)

    # NOTE This handles name clashes by escaping the first underscore
    return re.sub(_re_underscore, '__', parsed_name.name, count=1)


def pharmpy_to_nonmem(name: str):
    parsed_name = _parse_pharmpy_variable_name(name)

    if isinstance(parsed_name, ArrayItemName):
        return f'{parsed_name.array}({parsed_name.index})'

    if isinstance(parsed_name, MatrixItemName):
        return f'{parsed_name.matrix}({parsed_name.i},{parsed_name.j})'

    assert isinstance(parsed_name, VariableName)

    # NOTE Unescape the first underscore
    return re.sub(_re_underscore, '', parsed_name.name, count=1)


@dataclass(frozen=True)
class Name:
    pass


@dataclass(frozen=True)
class ArrayItemName(Name):
    array: str
    index: int


@dataclass(frozen=True)
class MatrixItemName(Name):
    matrix: str
    i: int
    j: int


@dataclass(frozen=True)
class VariableName(Name):
    name: str


_re_nonmem_array_name = re.compile(r'(THETA|ETA|EPS)\(([1-9]\d*)\)')
_re_nonmem_matrix_name = re.compile(r'(OMEGA|SIGMA)\(([1-9]\d*),([1-9]\d*)\)')


def _parse_nonmem_variable_name(name: str) -> Name:
    return _parse(name, _re_nonmem_array_name, _re_nonmem_matrix_name)


_re_pharmpy_array_name = re.compile(r'(THETA|ETA|EPS)_([1-9]\d*)')
_re_pharmpy_matrix_name = re.compile(r'(OMEGA|SIGMA)_([1-9]\d*)_([1-9]\d*)')


def _parse_pharmpy_variable_name(name: str) -> Name:
    return _parse(name, _re_pharmpy_array_name, _re_pharmpy_matrix_name)


def _parse(name: str, re_array: re.Pattern[str], re_matrix: re.Pattern[str]) -> Name:
    if match := re.fullmatch(re_array, name):
        array, index = match.groups()
        return ArrayItemName(array, int(index))
    if match := re.fullmatch(re_matrix, name):
        matrix, i, j = match.groups()
        return MatrixItemName(matrix, int(i), int(j))

    # NOTE Fallback
    return VariableName(name)
