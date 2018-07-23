"""List util for managing supported APIs"""
import importlib
import os.path
import pkgutil
from ..exceptions import ModelException
from collections import namedtuple


ModelAPI = namedtuple('API', 'module dir name')


class ModelAPIException(ModelException):
    pass


class ModelAPIList(list):
    def __init__(self, pkg_name):
        path = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
        apis = []
        for importer, modname, ispkg in pkgutil.iter_modules([path]):
            if ispkg and modname.startswith('api_'):
                module = importlib.import_module('.' + modname, pkg_name)
                apis += [ModelAPI(module, modname, module.name)]
        super().__init__(apis)


    def get(self, name):
        """Gets API from name

        Args:
            name: Name of API

        Returns:
            API match

        Raises:
            ModelAPIException: Name does nto match defined API
        """
        api = next((x for x in self if x.name == name), None)
        if not api:
            api_names = [x.name for x in self]
            raise ModelAPIException(
                "Undefined model API '%s' (defined names: %s)" %
                (name, ', '.join(["'%s'" % (n,) for n in api_names]))
            )
        return api


    def supports(self, lines):
        """Probes support for (raw code) lines and returns appropriate subset

        Args:
            lines: Code lines to use when probing for support

        Returns:
            ModelAPI(s) detecting support
        """
        detected = []
        for api in self:
            if api.module.detect(lines):
                detected += [api]
        return detected


    def filedetect(self, filename):
        """Detects appropriate model API for model file

        Args:
            filename: Filename (path) containing code to use for probing

        Returns:
            Unique ModelAPI reporting support

        Raises:
            ModelAPIException: None or multiple APIs reporting support
        """
        with open(filename, 'r') as f:
            content = f.read()
        lines = content.splitlines()
        apis = self.supports(lines)
        if len(apis) == 0:
            raise ModelAPIException("Unknown model filetype '%s'" % (filename,))
        elif len(apis) > 1:
            api_names = [x.name for x in apis]
            raise ModelAPIException(
                "Ambigiuos model filetype '%s' (multiple supporting APIs: %s)" %
                (filename, ', '.join(["'%s'" % (n,) for n in api_names]))
            )
        return apis[0]
