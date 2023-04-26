from collections.abc import Collection, Container, Iterable, Iterator, Sequence, Sized
from inspect import signature
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


def with_runtime_arguments_type_check(fn):
    sig = signature(fn)
    parameters = sig.parameters.values()
    type_hints = None

    def _wrapped(*args, **kwargs):
        # NOTE This delays loading annotations until first use
        nonlocal type_hints
        if type_hints is None:
            type_hints = get_type_hints(fn)

        for i, parameter in enumerate(parameters):
            if parameter.annotation is parameter.empty:
                # NOTE Do not check anything if there is no annotation
                continue

            name = parameter.name
            value = _arg(parameter, i, args, kwargs)

            # NOTE We cannot use parameter.annotation as this does not work
            # in combination with `from __future__ import annotations`.
            # See https://peps.python.org/pep-0563/#introducing-a-new-dictionary-for-the-string-literal-form-instead
            expected_types = _annotation_to_types(type_hints[name])
            if not any(map(lambda expected_type: _match(expected_type, value), expected_types)):
                raise TypeError(
                    f'Invalid `{parameter.name}`: got `{value}` of type {_value_type(value)},'
                    + (
                        f' expected {expected_types[0]}'
                        if len(expected_types) == 1
                        else f' expected one of {expected_types}.'
                    )
                )

        return fn(*args, **kwargs)

    return _wrapped


def _type(t):
    return type(None) if t is None else t


def _annotation_to_types(annotation):
    types = annotation if isinstance(annotation, list) else [annotation]
    return [_type(t) for t in types]


def _value_type(value):
    return (
        Type[value]  # pyright: ignore [reportGeneralTypeIssues]
        if isinstance(value, type)
        else type(value)
    )


def _kwargs(parameter, kwargs):
    try:
        return kwargs[parameter.name]
    except KeyError:
        default = parameter.default
        if default == parameter.empty:
            raise KeyError
        return default


def _arg(parameter, i, args, kwargs):
    if parameter.kind == parameter.POSITIONAL_ONLY:
        return args[i]
    elif parameter.kind == parameter.POSITIONAL_OR_KEYWORD:
        if i < len(args):
            return args[i]
        return _kwargs(parameter, kwargs)
    elif parameter.kind == parameter.VAR_POSITIONAL:
        raise NotImplementedError(parameter.kind)
    elif parameter.kind == parameter.KEYWORD_ONLY:
        return _kwargs(parameter, kwargs)
    else:
        assert parameter.kind == parameter.VAR_KEYWORD
        raise NotImplementedError(parameter.kind)


def _match_sequence_items(args, value):
    if args:
        assert len(args) == 1
        t = args[0]
        return all(map(lambda v: _match(t, v), value))
    else:
        return True


def _match(typing, value):
    origin = get_origin(typing)

    if origin is None:
        if typing is Any or typing is Optional:
            return True
        return isinstance(value, typing)

    if origin is Literal:
        # NOTE Empty literals return False
        return any(map(lambda t: value == t, get_args(typing)))

    if origin is list or origin is List:
        return isinstance(value, list) and _match_sequence_items(get_args(typing), value)

    if origin is Sequence:
        return isinstance(value, Sequence) and _match_sequence_items(get_args(typing), value)

    if origin is tuple or origin is Tuple:
        if not isinstance(value, tuple):
            return False

        args = get_args(typing)
        n = len(args)
        if n == 2 and args[1] is Ellipsis:
            return _match_sequence_items((args[0],), value)

        else:
            return len(value) == n and all(map(_match, args, value))

    if origin is Union:
        # NOTE Empty unions return False
        return any(map(lambda t: _match(t, value), get_args(typing)))

    if origin is Optional:
        args = get_args(typing)
        if args:
            assert len(args) == 1
            t = args[0]
            return value is None or _match(t, value)
        else:
            return True

    if origin is Sized:
        try:
            return isinstance(len(value), int)
        except TypeError:
            return False

    if origin is Container:
        # NOTE Cannot check value type because we do not know any candidate key
        return hasattr(value, '__contains__')

    if origin is Iterator:
        # NOTE Cannot check value type because we do not know any candidate key
        return hasattr(value, '__next__') and hasattr(value, '__iter__')

    if origin is Iterable:
        # NOTE Cannot check value type because of risk of side-effect
        try:
            iter(value)
            return True
        except TypeError:
            return False

    if origin is Collection:
        t = get_args(typing)
        return (
            _match(Sized, value)
            and _match(
                Container[t] if t else Container,  # pyright: ignore [reportGeneralTypeIssues]
                value,
            )
            and _match(
                Iterable[t] if t else Iterable,  # pyright: ignore [reportGeneralTypeIssues]
                value,
            )
        )

    raise NotImplementedError(origin)
