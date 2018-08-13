# -*- encoding: utf-8 -*-

"""Util for managing supported APIs."""

import importlib
import logging
import pkgutil
import sys


_MODULES = []


class ModelAPIException(Exception):
    pass


def init(path, pkg_name):
    log = logging.getLogger(__name__)
    prefix = pkg_name + '.'
    modules = dict()
    log.debug('searching APIs (path=%s, prefix=%s)' % (path, prefix))
    for importer, modname, ispkg in pkgutil.iter_modules(path, prefix):
        last = modname.rpartition('.')[-1]
        if last.startswith('api_'):
            spec = importer.find_spec(modname)
            if not spec:
                log.warning('no module spec, no import: %s' % (modname,))
            elif ispkg:
                module = importlib.import_module(modname, pkg_name)
            else:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            modules[last[4:]] = module
    if 'generic' not in modules:
        modnames = list(modules.keys())
        raise ModelAPIException(
            "generic API not found: %s" %
            (', '.join(modnames),)
        )
    setattr(sys.modules[__name__], '_MODULES', modules)
    return modules


def detectAPI(path):
    """Detects appropriate implementation from file Path"""
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    support = {k: mod for k, mod in _MODULES.items() if mod.detect(lines)}
    if len(support) == 0:
        raise ModelAPIException(
            "No modules (%s) detect support for file: %s" %
            (', '.join(list(_MODULES.keys())), path)
        )
    if len(support) > 1:
        raise ModelAPIException(
            "Multiple modules (%s) detects support for file: %s" %
            (', '.join(list(support.keys())), path)
        )
    module = list(support.values())[0]
    return module.Model


def getAPI(name='generic'):
    """Gets implementation from name"""
    if name not in _MODULES:
        raise ModelAPIException(
            "Module name '%s' is not defined (%s)" %
            (name, ', '.join(list(_MODULES.keys())))
        )
    return _MODULES[name]
