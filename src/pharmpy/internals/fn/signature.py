from functools import wraps
from inspect import Signature, signature
from typing import Callable


def with_same_arguments_as(ref: Callable):
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

    def _with_same_arguments(fn: Callable):
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

    return _with_same_arguments


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
