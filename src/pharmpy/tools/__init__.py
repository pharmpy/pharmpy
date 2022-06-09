from typing import Any, List

from .run import create_results, fit, read_results, retrieve_models, run_tool
from .run_amd import run_amd

__all__ = [
    'create_results',
    'fit',
    'read_results',
    'retrieve_models',
    'run_amd',
    'run_tool',
]

_dynamic__all__ = {
    'run_allometry',
    'run_covsearch',
    'run_iivsearch',
    'run_iovsearch',
    'run_modelsearch',
    'run_resmod',
}


def __getattr__(name: str) -> Any:
    try:
        if not name.startswith('run_') or name not in _dynamic__all__:
            raise AttributeError()

        tool_name = name[4:]
        from .wrap import wrap

        func = wrap(tool_name)
        globals()[name] = func  # NOTE This is only to avoid searching next time
        return func

    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_dir = sorted(set(dir()) | _dynamic__all__ | {'__dir__'})


def __dir__() -> List[str]:
    return _dir
