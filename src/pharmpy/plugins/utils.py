"""Utils for managing plugins."""

import importlib
import pkgutil
from pathlib import Path


class PluginError(Exception):
    pass


def detect_model(src):
    """Detects appropriate implementation from a source object
    Return a model object
    """

    if not hasattr(src, 'obj'):
        raise PluginError("Input to detect_model doesn't seem to be a source object")

    plugins = load_plugins()
    detected_classes = []
    for module in plugins:
        if hasattr(module, 'detect_model'):
            cls = module.detect_model(src)
            if cls:
                detected_classes.append(cls)

    if len(detected_classes) == 0:
        raise PluginError(f"No support for model {src.obj}")
    if len(detected_classes) > 1:
        raise PluginError(f"More than one model plugin supports model {src.obj}")
    return detected_classes[0]


def load_plugins():
    """Find and import all available plugins"""
    plugin_path = Path(__file__).resolve().parent

    modules = list()

    # str on path to workaround https://bugs.python.org/issue44061
    for _, modname, ispkg in pkgutil.iter_modules([str(plugin_path)], 'pharmpy.plugins.'):
        if ispkg:
            module = importlib.import_module(modname)
            modules.append(module)

    return modules
