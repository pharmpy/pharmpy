# -*- encoding: utf-8 -*-

"""Utils for managing plugins."""

import importlib
import pkgutil
from pathlib import Path


class PluginError(Exception):
    pass


def detect_model(obj):
    """Detects appropriate implementation from a general object
    The plugins can support paths to files but also any relevant object
    that could represent a model. Return a model object
    """

    plugins = load_plugins()
    detected_classes = [module.Model for module in plugins if hasattr(module, 'Model') and module.Model.detect(obj)]

    if len(detected_classes) == 0:
        raise PluginError(f"No support for model {obj}")
    if len(detected_classes) > 1:
        raise PluginError(f"More than one model plugin supports model {obj}")
    return detected_classes[0]


def load_plugins():
    """Find and import all available plugins
    """
    plugin_path = Path(__file__).resolve().parent

    modules = list()

    for _, modname, ispkg in pkgutil.iter_modules([plugin_path], 'pharmpy.plugins.'):
        if ispkg:
            module = importlib.import_module(modname)
            modules.append(module)

    return modules
