import os
import shutil
import time
import warnings
import weakref
from collections.abc import Collection, Container, Iterable, Iterator, Sequence, Sized
from functools import wraps
from inspect import Signature, signature
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable
from typing import Container as TypingContainer  # NOTE needed for Python 3.8
from typing import Iterable as TypingIterable  # NOTE needed for Python 3.8
from typing import List, Literal, Optional, Tuple, Type, Union, get_args, get_origin, get_type_hints

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.expressions import subs, sympify


class TemporaryDirectoryChanger:
    def __init__(self, path):
        self.path = path
        self.old_path = Path.cwd()

    def __enter__(self):
        os.chdir(self.path)
        return self

    def __exit__(self, *args):
        os.chdir(self.old_path)


if os.name == 'nt':
    # This is copied and modified from the python 3.9 implementation
    # The aim is to be able to handle Permission issues in Windows
    class TemporaryDirectory:
        """Create and return a temporary directory.  This has the same
        behavior as mkdtemp but can be used as a context manager.  For
        example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained
        in it are removed.
        """

        def __init__(self, suffix=None, prefix=None, dir=None):
            self.name = mkdtemp(suffix, prefix, dir)
            self._finalizer = weakref.finalize(
                self,
                self._cleanup,
                self.name,
                warn_message="Implicitly cleaning up {!r}".format(self),
            )

        @classmethod
        def _rmtree(cls, name):
            def onerror(func, path, exc_info):
                if issubclass(exc_info[0], PermissionError):

                    def resetperms(path):
                        try:
                            os.chflags(path, 0)
                        except AttributeError:
                            pass
                        os.chmod(path, 0o700)

                    try:
                        if path != name:
                            resetperms(os.path.dirname(path))
                        resetperms(path)

                        try:
                            os.unlink(path)
                        # PermissionError is raised on FreeBSD for directories
                        except (IsADirectoryError, PermissionError):
                            time.sleep(0.1)
                            cls._rmtree(path)
                    except FileNotFoundError:
                        pass
                elif issubclass(exc_info[0], FileNotFoundError):
                    pass
                else:
                    raise

            shutil.rmtree(name, onerror=onerror)

        @classmethod
        def _cleanup(cls, name, warn_message):
            cls._rmtree(name)
            warnings.warn(warn_message, ResourceWarning)

        def __repr__(self):
            return "<{} {!r}>".format(self.__class__.__name__, self.name)

        def __enter__(self):
            return self.name

        def __exit__(self, exc, value, tb):
            self.cleanup()

        def cleanup(self):
            if self._finalizer.detach():
                self._rmtree(self.name)

else:
    from tempfile import TemporaryDirectory  # noqa


_unit_subs = None


def unit_subs():

    global _unit_subs
    if _unit_subs is None:
        subs = {}
        import sympy.physics.units as units

        for k, v in units.__dict__.items():
            if isinstance(v, sympy.Expr) and v.has(units.Unit):
                subs[sympy.Symbol(k)] = v

        _unit_subs = subs

    return _unit_subs


def parse_units(s):
    return subs(sympify(s), unit_subs(), simultaneous=True) if isinstance(s, str) else s


def normalize_user_given_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path.expanduser()


def hash_df(df) -> int:
    values = pd.util.hash_pandas_object(df, index=True).values
    return hash(tuple(values))


def _signature_map(ref_signature: Signature):
    ref_args = []
    ref_defaults = {}
    for param in ref_signature.parameters.values():
        assert param.kind != param.VAR_POSITIONAL  # TODO handle varargs
        assert param.kind != param.VAR_KEYWORD  # TODO handle kwargs
        name = param.name
        ref_args.append(name)
        default = param.default
        if default is not param.empty:
            ref_defaults[name] = default

    return ref_args, ref_defaults


def same_arguments_as(ref: Callable):
    ref_signature = signature(ref)
    ref_args, ref_defaults = _signature_map(ref_signature)
    ref_args_set = set(ref_args)
    ref_index = {arg: i for i, arg in enumerate(ref_args)}

    def _lookup(args, kwargs, key):
        i = ref_index[key]
        if i < len(args):
            return args[i]
        try:
            return kwargs[key]
        except KeyError:
            return ref_defaults[key]

    def _with_same_signature(fn: Callable):

        fn_args, fn_defaults = _signature_map(signature(fn))

        assert set(fn_args) <= ref_args_set
        assert not fn_defaults

        @wraps(fn)
        def _wrapped(*args, **kwargs):

            if len(args) > len(ref_args):
                raise TypeError(
                    f'{fn.__name__}() takes {len(ref_args)} but {len(args)} where given'
                )

            for arg in kwargs:
                if arg not in ref_args_set:
                    raise TypeError(
                        f'{fn.__name__}() got an unexpected keyword argument: \'{arg}\''
                    )

            try:
                permuted = [_lookup(args, kwargs, arg) for arg in fn_args]
            except KeyError as e:
                arg = e.args[0]
                raise TypeError(
                    f'{fn.__name__}() missing 1 required positional argument: \'{arg}\''
                )

            return fn(*permuted)

        # NOTE Lazy annotations can only be resolved properly if the
        # declarations of `ref` and `fn` have the same locals() and
        # globals().
        _wrapped.__annotations__ = ref.__annotations__
        _wrapped.__signature__ = ref_signature

        return _wrapped

    return _with_same_signature


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
                TypingContainer[t] if t else Container,  # pyright: ignore [reportGeneralTypeIssues]
                value,
            )
            and _match(
                TypingIterable[t] if t else Iterable,  # pyright: ignore [reportGeneralTypeIssues]
                value,
            )
        )

    raise NotImplementedError(origin)


def runtime_type_check(fn):
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
