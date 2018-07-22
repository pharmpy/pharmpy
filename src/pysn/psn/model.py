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
    def __init__(self, filename):
        self.detect_type(filename)

    def __str__(self):
        return str(self.model)

    def detect_type(self, filename):
        self.type = None
        with open(filename, "r") as f:
            content = f.read()
        content_array = content.split("\n")

        for importer, modname, ispkg in pkgutil.iter_modules(models.__path__):
            if ispkg:
                modpath = models.__name__ + '.' + modname
                detect_module = importlib.import_module(modpath + '.detect')
                if detect_module.detect(content_array):
                    model_module = importlib.import_module(modpath + '.model')
                    self.type = modname
                    break

        self.model = model_module.Model(filename)
        self.input = self.model.input

        if self.type == None:
            raise ModelException("Unknown model type of file " + filename)
