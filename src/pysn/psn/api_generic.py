# -*- encoding: utf-8 -*-

from pathlib import Path


def detect(lines):
    return False


class ModelException(Exception):
    pass


class ModelParsingError(ModelException):
    def __init__(self, msg='model parse error'):
        super().__init__(msg)


class ModelLookupError(LookupError):
    def __init__(self, lookup):
        self.lookup = lookup
        try:
            reason = 'index %d out of range' % (self.lookup,)
        except TypeError:
            reason = 'name %s not found' % (repr(str(self.lookup)),)
        msg = 'submodel does not exist (%s)' % (reason,)
        super().__init__(msg)


class Model(object):
    """Generic model of any type

    Agnostic of model type. Subclass me to implement non-agnostic API."""

    def __init__(self, path, **kwargs):
        self.path = Path(path).resolve() if path else None
        if self.exists:
            self.load()

    @property
    def exists(self):
        if self.path and self.path.is_file():
            return True

    @property
    def content(self):
        if not self.exists:
            return None
        with open(self.path, 'r') as f:
            content = f.read()
        return content

    def write(self, *args, **kwargs):
        pass

    def validate(self):
        raise NotImplementedError

    def load(self):
        self.index = 0
        self.input = ModelInput(self)
        self.parameters = ParameterModel(self)

    @property
    def index(self):
        """Submodel unique index (setter might accept name lookup)"""
        return self._index

    @index.setter
    def index(self, new):
        if new != 0:
            raise ModelLookupError(new)
        self._index = new

    def __str__(self):
        if self.exists:
            return self.read


class ModelInput(object):
    """Implements API for :attr:`Model.input`, the model dataset"""
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """Gets the path of the dataset"""
        raise NotImplementedError


class ParameterModel:
    """Implements API for :attr:`Model.parameters`, the model parameters"""
    def __init__(self, model):
        self.model = model

    @property
    def initial_estimates(self):
        """Returns all initial estimates for model"""
        raise NotImplementedError
