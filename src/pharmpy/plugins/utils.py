"""Utils for managing plugins."""

from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules


class PluginError(Exception):
    pass


def detect_model(src):
    """Detects appropriate implementation from a source object
    Return a plugin module
    """

    plugins = load_plugins()
    detected_plugins = []
    for module in plugins:
        if hasattr(module, 'detect_model'):
            is_plugin = module.detect_model(src)
            if is_plugin:
                detected_plugins.append(module)

    if len(detected_plugins) == 0:
        raise PluginError(f"No support for model {src}")
    elif len(detected_plugins) > 1:
        raise PluginError(f"More than one model plugin supports model {src}")
    else:
        return detected_plugins[0]


def load_plugins():
    """Find and import all available plugins"""
    plugin_path = Path(__file__).resolve().parent

    # str on path to workaround https://bugs.python.org/issue44061
    plugin_path = str(plugin_path)

    return [
        import_module(modname)
        for _, modname, ispkg in iter_modules([plugin_path], 'pharmpy.plugins.')
        if ispkg
    ]
