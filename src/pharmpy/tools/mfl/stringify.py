from dataclasses import fields
from typing import Iterable, Tuple, Union

from .statement.feature.covariate import Ref
from .statement.feature.symbols import Name, Wildcard
from .statement.statement import Statement

StringifiableAtom = Union[int, str, Name, Wildcard, Ref]
Stringifiable = Union[StringifiableAtom, Tuple[StringifiableAtom, ...]]


def stringify(statements: Iterable[Statement]) -> str:
    return ';'.join(map(_stringify_statement, statements))


def _stringify_statement(statement: Statement) -> str:
    return (
        f'{_stringify_statement_type(statement)}'
        f'({",".join(map(_stringify_attribute, _filter_attributes(statement)))})'
    )


def _stringify_statement_type(statement: Statement) -> str:
    return statement.__class__.__name__.upper()


def _filter_attributes(statement: Statement) -> Iterable[Stringifiable]:
    it = iter(fields(statement))
    for field in it:
        value = getattr(statement, field.name)
        if value == field.default:
            break
        yield value

    for field in it:
        # FIXME Better handling
        value = getattr(statement, field.name)
        assert value == field.default


def _stringify_attribute(attribute: Stringifiable) -> str:
    if isinstance(attribute, Ref):
        return f'@{attribute.name}'
    elif isinstance(attribute, Name):
        return attribute.name
    elif isinstance(attribute, Wildcard):
        return '*'
    elif isinstance(attribute, str):
        return attribute
    elif isinstance(attribute, int):
        return str(attribute)
    elif isinstance(attribute, tuple):
        if len(attribute) == 1:
            return _stringify_attribute(attribute[0])
        elif len(attribute) >= 2:
            if type(i := attribute[0]) == type(j := attribute[-1]) == int:  # noqa E721
                assert isinstance(i, int)
                assert isinstance(j, int)
                if tuple(range(i, j + 1)) == attribute:
                    return f'{i}..{j}'

        return f'[{",".join(map(_stringify_attribute, attribute))}]'
    else:
        raise TypeError(type(attribute))
