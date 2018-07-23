import pkgutil
import importlib

from . import models

class ModelException(Exception):
    pass

class ModelParsingError(Exception):
    pass

class Model:
    """A generic model of any type
       will keep track of the native model class and the different apis
    """
    def __init__(self, filename=None):
        self.filename = filename
        if filename:
            self.detect_type()
        else:
            self.type = None

    @property
    def type(self):
        """Filetype (non-agnostic API) property"""
        return self._type

    @type.setter
    def type(self, value):
        if value:
            _ = get_api(value)
            self._type = value
        else:
            self._type = None

    def detect_type(self):
        with open(self.filename, 'r') as f:
            content = f.read()
        content_array = content.split('\n')

        for importer, modname, ispkg in pkgutil.iter_modules(models.__path__):
            if ispkg:
                api = get_api(modname)
                if api.detect(content_array):
                    self.type = modname
                    break

        if not self.type:
            raise ModelException("Unknown model type of file '%s'" % filename)
        else:
            self.init_model()

    def init_model(self):
        api = get_api(self.type)
        self.model = api.Model(self.filename)
        self.input = self.model.input

    def __str__(self):
        return str(self.model)

def get_api(name):
    modpath = models.__name__ + '.' + name
    api = importlib.import_module(modpath)
    if not api:
        raise ModelException("Unknown model type (no '%s' api module)" % name)
    return api
