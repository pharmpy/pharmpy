# -*- encoding: utf-8 -*-

"""Util for managing supported APIs."""

import importlib
import pkgutil
from pathlib import Path


class ModelAPIException(Exception):
    pass


def detectAPI(path):
    """Detects appropriate implementation from file path."""

    with open(str(path), 'r') as f:
        lines = f.read().splitlines()
    support = {k: mod for k, mod in _MODULES().items() if mod.detect(lines)}
    if len(support) == 0:
        raise ModelAPIException("None supports file '%s'. Modules: %s" % (path, _MODULES()))
    if len(support) > 1:
        raise ModelAPIException("Many supports file '%s'. Modules: %s" % (path, support))
    module = list(support.values())[0]
    return module.Model


def getAPI(name='generic'):
    """Gets implementation from name."""

    if name not in _MODULES():
        raise ModelAPIException(
            "Module name '%s' is not defined (%s)" %
            (name, ', '.join(list(_MODULES().keys())))
        )
    return _MODULES()[name]


def _MODULES():
    modules = dict()
    paths = [str(Path(__file__).resolve().parent)]

    for importer, modname, ispkg in pkgutil.iter_modules(paths, 'pysn.'):
        if modname == __name__:
            continue
        last = modname.rpartition('.')[-1]
        if last == 'generic' or last.startswith('api_'):
            spec = importer.find_spec(modname)
            if not spec:
                continue
            elif ispkg:
                module = importlib.import_module(modname, 'pysn')
            else:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            modules[last[4:] if last.startswith('api_') else last] = module
    if 'generic' not in modules:
        raise ModelAPIException("generic API not found: %s" % (modules,))
    return modules
