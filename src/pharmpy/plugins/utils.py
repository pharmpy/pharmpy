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

    plugins = load_plugins()
    detected_classes = []
    for module in plugins:
        if hasattr(module, 'detect_model'):
            cls = module.detect_model(src)
            if cls:
                detected_classes.append(cls)

    if len(detected_classes) == 0:
        raise PluginError(f"No support for model {src}")
    if len(detected_classes) > 1:
        raise PluginError(f"More than one model plugin supports model {src}")
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
