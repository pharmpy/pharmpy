"""Utils for managing detecting and parsing of external model types."""

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules


def detect_model(src):
    """Detects appropriate implementation from a source object
    Return an external model module
    """

    plugins = _load_external_modules()
    detected_modules = []
    for module in plugins:
        if hasattr(module, 'detect_model'):
            is_module = module.detect_model(src)
            if is_module:
                detected_modules.append(module)

    if len(detected_modules) == 0:
        raise TypeError(f"No support for model {src}")
    elif len(detected_modules) > 1:
        raise TypeError(f"More than one external model module supports model {src}")
    else:
        return detected_modules[0]


def _load_external_modules():
    """Find and import all available external modules"""
    path = Path(__file__).resolve().parent

    # str on path to workaround https://bugs.python.org/issue44061
    path = str(path)

    return [
        import_module(modname)
        for _, modname, ispkg in iter_modules([path], 'pharmpy.model.external.')
        if ispkg
    ]
